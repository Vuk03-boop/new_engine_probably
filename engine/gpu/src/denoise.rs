//! Phase 3E: the native reconstruction (`shaders/denoise.slang`), SVGF-style (Schied et al. 2017) on
//! the exact guides of ADR-0006.
//!
//! After the temporal pass, per frame: demodulate the resolved radiance by the material albedo and
//! estimate its variance (history moments, or the neighbours while the history is young), run
//! `levels` a-trous levels (steps 1, 2, 4, ...) that only mix pixels on the same surface (face,
//! material and integer plane) and stop at luminance edges scaled by the variance, then remodulate
//! into the shade radiance buffer, whose w (the temporal state) is kept. The history itself stays
//! unfiltered: the temporal pass's running mean is exact (3D) and the filter only shapes what is shown.
//!
//! [`DenoiseTargets`] owns the two frame-sized ping-pong buffers and the levels' luminance guide
//! and the compact surface keys (`GpuTemporal`, 44 B/px).

use std::mem::size_of;

use ash::vk;
use memory::Category;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::reference::RefMaterials;
use crate::reflect::{self, Field, Param};
use crate::temporal::History;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/denoise.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/denoise.json"));

/// The filter's sliders (3E record: swept, defaults at the measured plateau).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseSettings {
    /// A-trous levels (0: demodulate and remodulate only).
    pub levels: u32,
    /// Luminance edge-stopping strength in standard deviations (SVGF: 4).
    pub sigma_l: f32,
    /// The first `prefilter_levels` levels' edge-stopping compares 3x3 same-surface means of the level's
    /// input instead of the taps' own values (less energy loss at young ages); 0 is plain SVGF.
    pub prefilter_levels: u32,
    /// The history age from which the variance comes from the moments instead of the neighbours.
    pub moments_age: u32,
    /// 3G: the guide prefilter applies to pixels younger than this; older pixels compare their own
    /// (converged) values, so hard lighting edges are not blurred by 3x3 means. `u32::MAX` is the 3E
    /// filter (prefilter at every age).
    pub prefilter_age: u32,
    /// The edge-stopping variance is pre-blurred 3x3 on the same surface (SVGF); off saves 9 taps a level.
    pub variance_blur: bool,
}

impl Default for DenoiseSettings {
    fn default() -> DenoiseSettings {
        DenoiseSettings { levels: 2, sigma_l: 4.0, prefilter_levels: 1, moments_age: 8, variance_blur: true, prefilter_age: u32::MAX }
    }
}

/// Planted faults for negative controls; all off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DenoiseFaults {
    /// Mix every surface pixel regardless of its guide (blurs across edges and materials).
    pub ignore_guides: bool,
}

mod flags {
    pub const PREFILTER: u32 = 1;
    pub const INIT_GUIDE: u32 = 2;
    pub const REMODULATE: u32 = 4;
    pub const NO_VARIANCE_BLUR: u32 = 8;
    pub const FAULT_IGNORE_GUIDES: u32 = 256;
}

/// Host mirror of the shader's push constants (36 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub width: u32,
    pub height: u32,
    pub mode: u32,
    pub step: u32,
    pub src: u32,
    pub guide: u32,
    pub flags: u32,
    pub sigma_l: f32,
    pub moments_age: u32,
    pub prefilter_age: u32,
}

const _: () = assert!(size_of::<Params>() == 40);

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: std::mem::offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

pub fn host_layout() -> Vec<Param> {
    let names = ["radiance", "guides_0", "guides_1", "hist_0", "hist_1", "albedo", "filt_a", "filt_b", "lum_guide", "keys"];
    let mut v: Vec<Param> = names.iter().enumerate().map(|(i, n)| Param::Descriptor { name: n, binding: i as u64, element: vec![] }).collect();
    v.push(Param::PushConstants {
        name: "params",
        fields: vec![
            field!(Params, width),
            field!(Params, height),
            field!(Params, mode),
            field!(Params, step),
            field!(Params, src),
            field!(Params, guide),
            field!(Params, flags),
            field!(Params, sigma_l),
            field!(Params, moments_age),
            field!(Params, prefilter_age),
        ],
    });
    v
}

/// The filter's ping-pong buffers and luminance guide for one frame size.
pub struct DenoiseTargets {
    pub width: u32,
    pub height: u32,
    filt: [Buffer; 2],
    lum: Buffer,
    keys: Buffer,
}

impl DenoiseTargets {
    /// All-or-nothing.
    pub fn new(gpu: &Gpu, alloc: &mut Allocator, width: u32, height: u32) -> Result<DenoiseTargets> {
        let px = (width * height) as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let mut made: Vec<Buffer> = Vec::new();
        for bytes in [px * 16, px * 16, px * 4, px * 8] {
            match alloc.create_buffer(gpu, bytes, usage, Category::GpuTemporal, Kind::Device) {
                Ok(b) => made.push(b),
                Err(e) => {
                    for b in made {
                        alloc.free(gpu, b);
                    }
                    return Err(e);
                }
            }
        }
        let keys = made.pop().unwrap();
        let lum = made.pop().unwrap();
        let b = made.pop().unwrap();
        let a = made.pop().unwrap();
        Ok(DenoiseTargets { width, height, filt: [a, b], lum, keys })
    }

    pub fn device_bytes(&self) -> u64 {
        self.filt.iter().map(|b| b.size).sum::<u64>() + self.lum.size + self.keys.size
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        for b in self.filt {
            alloc.free(gpu, b);
        }
        alloc.free(gpu, self.lum);
        alloc.free(gpu, self.keys);
    }
}

pub struct DenoiseBindings {
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl DenoiseBindings {
    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }
}

pub struct Denoise {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Denoise {
    pub fn new(gpu: &Gpu) -> Result<Denoise> {
        Self::with_reflection(gpu, REFLECTION)
    }

    pub fn with_reflection(gpu: &Gpu, reflection: &str) -> Result<Denoise> {
        reflect::check(reflection, &host_layout()).map_err(GpuError::Layout)?;
        let dev = &gpu.device;
        let bindings: Vec<_> = (0..10).map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
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
        Ok(Denoise { set_layout, layout, pipeline })
    }

    /// Binds the shade radiance, the history's guides and moments, the albedo table and `targets`.
    /// Rebind after a resize (new history and targets) or a new albedo table.
    pub fn bind(&self, gpu: &Gpu, radiance: &Buffer, history: &History, albedo: &RefMaterials, targets: &DenoiseTargets) -> Result<DenoiseBindings> {
        let dev = &gpu.device;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 10 }];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        let layouts = [self.set_layout];
        let set = match unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.vk("vkAllocateDescriptorSets") {
            Ok(s) => s[0],
            Err(e) => {
                unsafe { dev.destroy_descriptor_pool(pool, None) };
                return Err(e);
            }
        };
        let whole = |b: &Buffer| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let (g, h) = (history.guides(), history.hist());
        let bufs = [
            whole(radiance),
            whole(&g[0]),
            whole(&g[1]),
            whole(&h[0]),
            whole(&h[1]),
            [vk::DescriptorBufferInfo { buffer: albedo.buffer.buffer, offset: 0, range: (albedo.count as u64 * 16).max(16) }],
            whole(&targets.filt[0]),
            whole(&targets.filt[1]),
            whole(&targets.lum),
            whole(&targets.keys),
        ];
        let writes: Vec<_> = bufs.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
        Ok(DenoiseBindings { pool, set })
    }

    /// Records the filter after the temporal pass of this frame (which `history` has just advanced
    /// past). The caller makes the radiance visible to its consumer (`shade::radiance_to_fragment`).
    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, bindings: &DenoiseBindings, history: &History, targets: &DenoiseTargets, s: DenoiseSettings, faults: DenoiseFaults) {
        assert_eq!((history.width, history.height), (targets.width, targets.height), "history and denoise targets must have the same size");
        let dev = &gpu.device;
        let barrier = || {
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
            unsafe { dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
        };
        let base = Params {
            width: targets.width,
            height: targets.height,
            guide: history.parity() as u32,
            flags: if faults.ignore_guides { flags::FAULT_IGNORE_GUIDES } else { 0 } | if s.variance_blur { 0 } else { flags::NO_VARIANCE_BLUR },
            sigma_l: s.sigma_l,
            moments_age: s.moments_age,
            prefilter_age: s.prefilter_age,
            ..Params::default()
        };
        let dispatch = |p: Params| {
            let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
            barrier();
            unsafe {
                dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
                dev.cmd_dispatch(cmd, targets.width.div_ceil(8), targets.height.div_ceil(8), 1);
            }
        };
        unsafe {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[bindings.set], &[]);
        }
        // The first level's guide comes from init, and the last level remodulates (no extra passes).
        let init_guide = s.levels > 0 && s.prefilter_levels > 0;
        dispatch(Params { mode: 0, flags: base.flags | if init_guide { flags::INIT_GUIDE } else { 0 }, ..base });
        for level in 0..s.levels {
            let pre = level < s.prefilter_levels;
            if pre && level > 0 {
                dispatch(Params { mode: 3, src: level % 2, ..base });
            }
            let mut flags = base.flags | if pre { flags::PREFILTER } else { 0 };
            if level + 1 == s.levels {
                flags |= flags::REMODULATE;
            }
            dispatch(Params { mode: 1, step: 1 << level, src: level % 2, flags, ..base });
        }
        if s.levels == 0 {
            dispatch(Params { mode: 2, src: 0, ..base });
        }
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
