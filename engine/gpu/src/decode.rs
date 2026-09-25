//! GPU decode of uploaded region meshes (`shaders/mesh_decode.slang`): the executed half of the
//! host/shader layout check. The pipeline is created only after the module's reflection matches
//! the host declaration below ([`crate::reflect::check`]).
//!
//! Every quad record is decoded, and every triangle is checked on the device: its three vertices
//! lie on its quad (`tri_quad`), and its winding faces the quad's outward normal.

use std::mem::{offset_of, size_of};

use ash::vk;
use memory::Category;

use crate::alloc::{Allocator, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::mesh::GpuMeshes;
use crate::reflect::{self, Field, Param};
use crate::submit::{all_to_host, Submitter};
use crate::timeline::Timeline;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh_decode.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/mesh_decode.json"));

/// Host mirror of the shader's `DecodedQuad`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodedQuad {
    pub material: u32,
    pub face: u32,
    pub plane: u32,
    pub u0: u32,
    pub v0: u32,
    pub u1: u32,
    pub v1: u32,
    pub pad: u32,
}

/// Host mirror of the shader's `TriCheck`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TriCheck {
    pub quad: u32,
    pub ok: u32,
}

/// Every bit of [`TriCheck::ok`]: three vertices on the quad, quad in range, outward winding.
pub const TRI_OK: u32 = 0x1F;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub quad_count: u32,
    pub tri_count: u32,
}

macro_rules! fields {
    ($t:ty: $($f:ident),*) => { vec![$(Field { name: stringify!($f), offset: offset_of!($t, $f) as u64, size: 4 }),*] };
}

/// What the host code above assumes about the shader.
pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "quads", binding: 0, element: vec![] },
        Param::Descriptor { name: "vertices", binding: 1, element: vec![] },
        Param::Descriptor { name: "indices", binding: 2, element: vec![] },
        Param::Descriptor { name: "tri_quad", binding: 3, element: vec![] },
        Param::Descriptor { name: "decoded", binding: 4, element: fields!(DecodedQuad: material, face, plane, u0, v0, u1, v1, pad) },
        Param::Descriptor { name: "tri_checks", binding: 5, element: fields!(TriCheck: quad, ok) },
        Param::PushConstants { name: "params", fields: fields!(Params: quad_count, tri_count) },
    ]
}

/// One region's decode results.
#[derive(Clone, Debug, Default)]
pub struct Decoded {
    pub quads: Vec<DecodedQuad>,
    pub triangles: Vec<TriCheck>,
}

pub struct Decoder {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

const BINDINGS: u32 = 6;

impl Decoder {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        Self::with_reflection(gpu, REFLECTION)
    }

    /// Checks `reflection` against [`host_layout`] first; a mismatch is `GpuError::Layout` and no
    /// Vulkan object is created.
    pub fn with_reflection(gpu: &Gpu, reflection: &str) -> Result<Self> {
        reflect::check(reflection, &host_layout()).map_err(GpuError::Layout)?;
        let dev = &gpu.device;
        let bindings: Vec<_> = (0..BINDINGS)
            .map(|b| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = unsafe { dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.vk("vkCreateDescriptorSetLayout")?;
        let pc = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(size_of::<Params>() as u32)];
        let sl = [set_layout];
        let layout = unsafe { dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&pc), None) }.vk("vkCreatePipelineLayout")?;
        let (chunks, rest) = SPIRV.as_chunks::<4>();
        assert!(rest.is_empty(), "SPIR-V is whole words");
        let words: Vec<u32> = chunks.iter().map(|&c| u32::from_le_bytes(c)).collect();
        let module = unsafe { dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }.vk("vkCreateShaderModule")?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let pipeline = unsafe { dev.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
        unsafe { dev.destroy_shader_module(module, None) };
        let pipeline = pipeline.map_err(|(_, e)| GpuError::Vk { call: "vkCreateComputePipelines", result: e })?[0];
        Ok(Self { set_layout, layout, pipeline })
    }

    /// Decodes every region on the GPU and returns the results per region, in region order.
    /// Waits for earlier uploads through queue order.
    pub fn run(&self, gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline, meshes: &GpuMeshes) -> Result<Vec<Decoded>> {
        let dev = &gpu.device;
        let align = meshes.align;
        let (qs, ts) = (size_of::<DecodedQuad>() as u64, size_of::<TriCheck>() as u64);
        // Output layout: per region, decoded quads then triangle checks, each aligned.
        let mut offsets = Vec::new();
        let mut total = 0u64;
        for r in meshes.regions.values() {
            let q = total;
            let t = (q + r.quad_count * qs).next_multiple_of(align);
            total = (t + r.triangle_count * ts).next_multiple_of(align).max(t + align);
            offsets.push((q, t));
        }
        let mut out = alloc.create_buffer(gpu, total.max(align), vk::BufferUsageFlags::STORAGE_BUFFER, Category::Staging, Kind::Host)?;
        out.mapped().unwrap().fill(0xAB);
        let n = meshes.regions.len().max(1) as u32;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: BINDINGS * n }];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        let layouts = vec![self.set_layout; n as usize];
        let sets = unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.vk("vkAllocateDescriptorSets")?;

        let mut sub = Submitter::new(gpu)?;
        let cmd = sub.begin(gpu, timeline)?;
        unsafe { dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline) };
        for ((r, &set), &(qo, to)) in meshes.regions.values().zip(&sets).zip(&offsets) {
            let s = r.sections;
            let b = r.buffer.buffer;
            let infos = [
                [vk::DescriptorBufferInfo { buffer: b, offset: s.quads, range: (r.quad_count * 8).max(8) }],
                [vk::DescriptorBufferInfo { buffer: b, offset: s.vertices, range: (r.vertex_count * 8).max(8) }],
                [vk::DescriptorBufferInfo { buffer: b, offset: s.indices, range: (r.triangle_count * 12).max(4) }],
                [vk::DescriptorBufferInfo { buffer: b, offset: s.tri_quad, range: (r.triangle_count * 4).max(4) }],
                [vk::DescriptorBufferInfo { buffer: out.buffer, offset: qo, range: (r.quad_count * qs).max(qs) }],
                [vk::DescriptorBufferInfo { buffer: out.buffer, offset: to, range: (r.triangle_count * ts).max(ts) }],
            ];
            let writes: Vec<_> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
            unsafe { dev.update_descriptor_sets(&writes, &[]) };
            let p = Params { quad_count: r.quad_count as u32, tri_count: r.triangle_count as u32 };
            let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
            unsafe {
                dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[set], &[]);
                dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
                dev.cmd_dispatch(cmd, (r.quad_count.max(r.triangle_count) as u32).div_ceil(64), 1, 1);
            }
        }
        all_to_host(gpu, cmd);
        let v = sub.submit(gpu, timeline, cmd, &[])?;
        timeline.wait(gpu, v, u64::MAX)?;
        let m = out.mapped_ref().unwrap();
        let word = |o: usize| u32::from_le_bytes(m[o..o + 4].try_into().unwrap());
        let result = meshes
            .regions
            .values()
            .zip(&offsets)
            .map(|(r, &(qo, to))| Decoded {
                quads: (0..r.quad_count as usize)
                    .map(|i| {
                        let o = qo as usize + i * qs as usize;
                        let w: Vec<u32> = (0..8).map(|k| word(o + 4 * k)).collect();
                        DecodedQuad { material: w[0], face: w[1], plane: w[2], u0: w[3], v0: w[4], u1: w[5], v1: w[6], pad: w[7] }
                    })
                    .collect(),
                triangles: (0..r.triangle_count as usize)
                    .map(|i| {
                        let o = to as usize + i * ts as usize;
                        TriCheck { quad: word(o), ok: word(o + 4) }
                    })
                    .collect(),
            })
            .collect();
        sub.destroy(gpu);
        unsafe { dev.destroy_descriptor_pool(pool, None) };
        alloc.free(gpu, out);
        Ok(result)
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
