//! Phase 2C-1: raster primary visibility into the ADR-0003 G-buffer, without a window.
//!
//! Conventions (2C decision 5):
//! - World: right-handed, Y up, voxel units. The camera looks along `forward`.
//! - Camera-relative: each region is drawn with `region origin − camera origin` (integers, exact in
//!   f32), where the camera origin is `floor(eye)`. The fractional eye position is folded into the
//!   view-projection rows, computed in f64 on the host.
//! - Reverse-Z infinite projection: clip z = `near`, clip w = distance along `forward`, so
//!   depth = near / distance. Depth clears to 0 and the test is `GREATER`.
//! - Vulkan's Y-down framebuffer is handled with a negative viewport height; front faces are then
//!   counter-clockwise, which matches ADR-0004's winding (counter-clockwise seen from outside).
//!   Back faces are culled.
//! - Pixel centres (2C decision 6): pixel `(x, y)` samples at `(x + 0.5, y + 0.5)`; no MSAA.
//!   [`Camera::ray`] gives the matching CPU ray.
//!
//! Targets live under `Category::GpuTemporal` ("frame targets and temporal histories").

use std::mem::{offset_of, size_of};

use ash::vk;
use memory::Category;
use world::reference::Ray;

use crate::alloc::{Allocator, Image, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::mesh::GpuMeshes;
use crate::reflect::{self, Field, Param, Varying};
use crate::submit::{all_to_host, Submitter};
use crate::timeline::Timeline;

pub const VS_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/raster_vs.spv"));
pub const VS_REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/raster_vs.json"));
pub const FS_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/raster_fs.spv"));
pub const FS_REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/raster_fs.json"));

pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
/// Colour targets in location order: (shader output name, format, reflected scalar type, components).
pub const COLOR_TARGETS: [(&str, vk::Format, &str, u64); 3] = [
    ("normal", vk::Format::R16G16_SNORM, "float32", 2),
    ("material", vk::Format::R16_UINT, "uint32", 1),
    ("surface", vk::Format::R32G32_UINT, "uint32", 2),
];
/// Vertex input: ADR-0004 positions.
pub const VERTEX_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Cleared values where no surface was drawn.
pub const BACKGROUND_MATERIAL: u16 = u16::MAX;
pub const BACKGROUND_SURFACE: [u32; 2] = [u32::MAX, u32::MAX];

/// A pinhole camera in voxel units.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: [f64; 3],
    /// Orthonormal basis.
    pub forward: [f64; 3],
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub tan_half_x: f64,
    pub tan_half_y: f64,
    pub near: f64,
    pub width: u32,
    pub height: u32,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

impl Camera {
    /// Looks from `eye` at `target` (voxel units) with world Y up.
    pub fn look_at(eye: [f64; 3], target: [f64; 3], vertical_fov_deg: f64, width: u32, height: u32, near: f64) -> Camera {
        let forward = normalize(sub(target, eye));
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        let tan_half_y = (vertical_fov_deg.to_radians() / 2.0).tan();
        Camera { eye, forward, right, up, tan_half_x: tan_half_y * width as f64 / height as f64, tan_half_y, near, width, height }
    }

    /// The integer camera origin that regions are drawn relative to.
    pub fn origin(&self) -> [i32; 3] {
        self.eye.map(|e| e.floor() as i32)
    }

    /// Direction through framebuffer point `(fx, fy)` (pixel `(x, y)`'s centre is `(x + 0.5, y + 0.5)`).
    /// Its `forward` component is exactly 1, so the ray parameter `t` equals the view distance.
    pub fn dir(&self, fx: f64, fy: f64) -> [f64; 3] {
        let sx = (2.0 * fx / self.width as f64 - 1.0) * self.tan_half_x;
        let sy = (1.0 - 2.0 * fy / self.height as f64) * self.tan_half_y;
        [0, 1, 2].map(|a| self.forward[a] + sx * self.right[a] + sy * self.up[a])
    }

    /// The CPU ray for pixel `(x, y)`: through its centre.
    pub fn ray(&self, x: u32, y: u32) -> Ray {
        Ray { origin: self.eye, dir: self.dir(x as f64 + 0.5, y as f64 + 0.5) }
    }

    /// View-projection rows for camera-relative positions (world − [`Camera::origin`]).
    pub fn rows(&self) -> [[f32; 4]; 4] {
        let o = self.origin();
        let f = [self.eye[0] - o[0] as f64, self.eye[1] - o[1] as f64, self.eye[2] - o[2] as f64];
        let row = |v: [f64; 3], s: f64| [(v[0] / s) as f32, (v[1] / s) as f32, (v[2] / s) as f32, (-dot(v, f) / s) as f32];
        [row(self.right, self.tan_half_x), row(self.up, self.tan_half_y), [0.0, 0.0, 0.0, self.near as f32], row(self.forward, 1.0)]
    }

    /// View distance of a reverse-Z depth value.
    pub fn distance(&self, depth: f32) -> f64 {
        self.near / depth as f64
    }
}

/// Host mirror of the shader's `Params` push constants.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub row0: [f32; 4],
    pub row1: [f32; 4],
    pub row2: [f32; 4],
    pub row3: [f32; 4],
    pub region_offset: [f32; 4],
    pub region_index: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

/// What the host code assumes about both raster modules.
pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "quads", binding: 0, element: vec![] },
        Param::Descriptor { name: "tri_quad", binding: 1, element: vec![] },
        Param::PushConstants {
            name: "params",
            fields: vec![
                field!(Params, row0),
                field!(Params, row1),
                field!(Params, row2),
                field!(Params, row3),
                field!(Params, region_offset),
                field!(Params, region_index),
                field!(Params, pad0),
                field!(Params, pad1),
                field!(Params, pad2),
            ],
        },
    ]
}

/// The vertex input and colour outputs the pipeline below is built for.
pub fn host_varyings() -> (Vec<Varying>, Vec<Varying>) {
    let v = |name: &str, location: u64, scalar: &str, components: u64| Varying { name: name.into(), location, scalar: scalar.into(), components };
    let inputs = vec![v("position", 0, "float32", 4)];
    let outputs = COLOR_TARGETS.iter().enumerate().map(|(i, &(n, _, s, c))| v(n, i as u64, s, c)).collect();
    (inputs, outputs)
}

/// Checks both modules' reflection against the host declarations.
pub fn check_reflection(vs: &str, fs: &str) -> std::result::Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for (stage, json) in [("vertex", vs), ("fragment", fs)] {
        if let Err(e) = reflect::check(json, &host_layout()) {
            errors.extend(e.into_iter().map(|e| format!("{stage}: {e}")));
        }
    }
    let (want_in, want_out) = host_varyings();
    match (reflect::varyings(vs), reflect::varyings(fs)) {
        (Ok((vin, _)), Ok((_, fout))) => {
            if vin != want_in {
                errors.push(format!("vertex inputs {vin:?}, host expects {want_in:?}"));
            }
            if fout != want_out {
                errors.push(format!("fragment outputs {fout:?}, host expects {want_out:?}"));
            }
        }
        (a, b) => errors.push(format!("varyings: {:?} {:?}", a.err(), b.err())),
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// The ADR-0003 primary-visibility targets.
pub struct Targets {
    pub extent: vk::Extent2D,
    pub depth: Image,
    /// In [`COLOR_TARGETS`] order: normal, material, surface.
    pub color: [Image; 3],
}

impl Targets {
    /// All-or-nothing: on a refused grant, the images made so far are freed.
    pub fn new(gpu: &Gpu, alloc: &mut Allocator, extent: vk::Extent2D) -> Result<Targets> {
        // SAMPLED: the 2C-2 debug views read the targets.
        let color_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED;
        let depth = alloc.create_image(gpu, DEPTH_FORMAT, extent, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED, vk::ImageAspectFlags::DEPTH, Category::GpuTemporal)?;
        let mut color = Vec::new();
        for &(_, format, _, _) in &COLOR_TARGETS {
            match alloc.create_image(gpu, format, extent, color_usage, vk::ImageAspectFlags::COLOR, Category::GpuTemporal) {
                Ok(i) => color.push(i),
                Err(e) => {
                    alloc.free_image(gpu, depth);
                    for i in color {
                        alloc.free_image(gpu, i);
                    }
                    return Err(e);
                }
            }
        }
        let color: [Image; 3] = color.try_into().expect("three colour targets");
        Ok(Targets { extent, depth, color })
    }

    /// Device bytes the driver required for the four images.
    pub fn device_bytes(&self) -> u64 {
        self.depth.alloc_size() + self.color.iter().map(Image::alloc_size).sum::<u64>()
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free_image(gpu, self.depth);
        for i in self.color {
            alloc.free_image(gpu, i);
        }
    }
}

/// Planted faults for negative controls. `Default` draws correctly.
#[derive(Clone, Copy, Debug, Default)]
pub struct Faults {
    /// Added to every region's offset (a camera-offset bug).
    pub offset_error: [i32; 3],
    /// Region (by draw index) left out (a missing-region bug).
    pub skip_region: Option<usize>,
}

/// The G-buffer read back to the host, row-major from the top-left pixel.
#[derive(Clone, Debug)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub depth: Vec<f32>,
    pub normal: Vec<[i16; 2]>,
    pub material: Vec<u16>,
    pub surface: Vec<[u32; 2]>,
}

/// Descriptor sets for one set of uploaded meshes: set `i` binds region `i`'s quad table.
pub struct Bindings {
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl Bindings {
    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_descriptor_pool(self.pool, None) };
    }

    /// The pool that owns the sets, for deferred destruction (2E retirement).
    pub fn into_pool(self) -> vk::DescriptorPool {
        self.pool
    }
}

pub struct Raster {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

fn module(gpu: &Gpu, spirv: &[u8]) -> Result<vk::ShaderModule> {
    let (chunks, rest) = spirv.as_chunks::<4>();
    assert!(rest.is_empty(), "SPIR-V is whole words");
    let words: Vec<u32> = chunks.iter().map(|&c| u32::from_le_bytes(c)).collect();
    unsafe { gpu.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }.vk("vkCreateShaderModule")
}

impl Raster {
    /// The pipeline with the convention's front face (counter-clockwise).
    pub fn new(gpu: &Gpu) -> Result<Raster> {
        Self::with_front_face(gpu, vk::FrontFace::COUNTER_CLOCKWISE, VS_REFLECTION, FS_REFLECTION)
    }

    /// `front` other than counter-clockwise is only for the planted-winding control. The
    /// reflection is checked first; a mismatch is `GpuError::Layout` and no Vulkan object is created.
    pub fn with_front_face(gpu: &Gpu, front: vk::FrontFace, vs_reflection: &str, fs_reflection: &str) -> Result<Raster> {
        check_reflection(vs_reflection, fs_reflection).map_err(GpuError::Layout)?;
        let dev = &gpu.device;
        let b = |i: u32| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let binding = [b(0), b(1)];
        let set_layout = unsafe { dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding), None) }.vk("vkCreateDescriptorSetLayout")?;
        let pc = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).offset(0).size(size_of::<Params>() as u32)];
        let sl = [set_layout];
        let layout = unsafe { dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&pc), None) }.vk("vkCreatePipelineLayout")?;

        let vs = module(gpu, VS_SPIRV)?;
        let fs = module(gpu, FS_SPIRV)?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"main"),
        ];
        let vb = [vk::VertexInputBindingDescription { binding: 0, stride: 8, input_rate: vk::VertexInputRate::VERTEX }];
        let va = [vk::VertexInputAttributeDescription { location: 0, binding: 0, format: VERTEX_FORMAT, offset: 0 }];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&vb).vertex_attribute_descriptions(&va);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::BACK).front_face(front).line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default().depth_test_enable(true).depth_write_enable(true).depth_compare_op(vk::CompareOp::GREATER);
        let attachments = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA); 3];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = COLOR_TARGETS.map(|t| t.1);
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats).depth_attachment_format(DEPTH_FORMAT);
        let info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering)];
        let pipeline = unsafe { dev.create_graphics_pipelines(vk::PipelineCache::null(), &info, None) };
        unsafe {
            dev.destroy_shader_module(vs, None);
            dev.destroy_shader_module(fs, None);
        }
        let pipeline = pipeline.map_err(|(_, e)| GpuError::Vk { call: "vkCreateGraphicsPipelines", result: e })?[0];
        Ok(Raster { set_layout, layout, pipeline })
    }

    /// One descriptor set per region, binding its quad table and triangle-quad map.
    pub fn bind(&self, gpu: &Gpu, meshes: &GpuMeshes) -> Result<Bindings> {
        let dev = &gpu.device;
        let n = meshes.regions.len().max(1) as u32;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 * n }];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        let layouts = vec![self.set_layout; meshes.regions.len()];
        let sets = if layouts.is_empty() {
            Vec::new()
        } else {
            unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)) }.vk("vkAllocateDescriptorSets")?
        };
        for (r, &set) in meshes.regions.values().zip(&sets) {
            let quads = [vk::DescriptorBufferInfo { buffer: r.buffer.buffer, offset: r.sections.quads, range: (r.quad_count * 8).max(8) }];
            let tris = [vk::DescriptorBufferInfo { buffer: r.buffer.buffer, offset: r.sections.tri_quad, range: (r.triangle_count * 4).max(4) }];
            let w = [
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&quads),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&tris),
            ];
            unsafe { dev.update_descriptor_sets(&w, &[]) };
        }
        Ok(Bindings { pool, sets })
    }

    /// Records the G-buffer pass: layouts to attachment, clear, one indexed draw per region.
    /// Leaves the targets in their attachment layouts.
    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, targets: &Targets, meshes: &GpuMeshes, bindings: &Bindings, camera: &Camera, faults: Faults) {
        let dev = &gpu.device;
        let range = |aspect| vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        let mut to_attachment: Vec<_> = targets
            .color
            .iter()
            .map(|i| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(i.image)
                    .subresource_range(range(vk::ImageAspectFlags::COLOR))
            })
            .collect();
        to_attachment.push(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image(targets.depth.image)
                .subresource_range(range(vk::ImageAspectFlags::DEPTH)),
        );
        unsafe { dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_attachment)) };

        let clears = [
            vk::ClearColorValue { float32: [0.0; 4] },
            vk::ClearColorValue { uint32: [BACKGROUND_MATERIAL as u32, 0, 0, 0] },
            vk::ClearColorValue { uint32: [BACKGROUND_SURFACE[0], BACKGROUND_SURFACE[1], 0, 0] },
        ];
        let color: Vec<_> = targets
            .color
            .iter()
            .zip(clears)
            .map(|(i, c)| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(i.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: c })
            })
            .collect();
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(targets.depth.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } });
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: targets.extent };
        let info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&color).depth_attachment(&depth);
        let (w, h) = (targets.extent.width as f32, targets.extent.height as f32);
        let viewport = [vk::Viewport { x: 0.0, y: h, width: w, height: -h, min_depth: 0.0, max_depth: 1.0 }];
        let rows = camera.rows();
        let origin = camera.origin();
        unsafe {
            dev.cmd_begin_rendering(cmd, &info);
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            dev.cmd_set_viewport(cmd, 0, &viewport);
            dev.cmd_set_scissor(cmd, 0, &[area]);
        }
        for (i, (r, &set)) in meshes.regions.values().zip(&bindings.sets).enumerate() {
            if faults.skip_region == Some(i) || r.triangle_count == 0 {
                continue;
            }
            let o = r.key.origin(meshes.size);
            let offset = [o.x - origin[0] + faults.offset_error[0], o.y - origin[1] + faults.offset_error[1], o.z - origin[2] + faults.offset_error[2]];
            let p = Params { row0: rows[0], row1: rows[1], row2: rows[2], row3: rows[3], region_offset: [offset[0] as f32, offset[1] as f32, offset[2] as f32, 0.0], region_index: i as u32, ..Params::default() };
            let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
            unsafe {
                dev.cmd_bind_vertex_buffers(cmd, 0, &[r.buffer.buffer], &[r.sections.vertices]);
                dev.cmd_bind_index_buffer(cmd, r.buffer.buffer, r.sections.indices, vk::IndexType::UINT32);
                dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[set], &[]);
                dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytes);
                dev.cmd_draw_indexed(cmd, (3 * r.triangle_count) as u32, 1, 0, 0, 0);
            }
        }
        unsafe { dev.cmd_end_rendering(cmd) };
    }

    /// Renders and reads the four targets back to the host, waiting for completion.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_host(&self, gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline, targets: &Targets, meshes: &GpuMeshes, bindings: &Bindings, camera: &Camera, faults: Faults) -> Result<Frame> {
        let dev = &gpu.device;
        let (w, h) = (targets.extent.width, targets.extent.height);
        assert_eq!((w, h), (camera.width, camera.height), "camera and targets must have the same size");
        let px = (w * h) as u64;
        // Readback sections: depth 4 B, normal 4 B, material 2 B, surface 8 B per pixel.
        let sizes = [4 * px, 4 * px, 2 * px, 8 * px];
        let mut offsets = [0u64; 4];
        let mut total = 0;
        for (o, s) in offsets.iter_mut().zip(sizes) {
            *o = total;
            total = (total + s).next_multiple_of(16);
        }
        let tmp = alloc.create_buffer(gpu, total, vk::BufferUsageFlags::TRANSFER_DST, Category::Staging, Kind::Host)?;
        let mut sub = Submitter::new(gpu)?;
        let result = (|| {
            let cmd = sub.begin(gpu, timeline)?;
            self.record(gpu, cmd, targets, meshes, bindings, camera, faults);
            let range = |aspect| vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            let images: [(&Image, vk::ImageLayout, vk::PipelineStageFlags2, vk::AccessFlags2); 4] = [
                (&targets.depth, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE),
                (&targets.color[0], vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
                (&targets.color[1], vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
                (&targets.color[2], vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
            ];
            let to_src: Vec<_> = images
                .iter()
                .map(|&(i, layout, stage, access)| {
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(stage)
                        .src_access_mask(access)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .old_layout(layout)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .image(i.image)
                        .subresource_range(range(i.aspect))
                })
                .collect();
            unsafe { dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_src)) };
            for (&(i, ..), &off) in images.iter().zip(&offsets) {
                let region = [vk::BufferImageCopy {
                    buffer_offset: off,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers { aspect_mask: i.aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
                }];
                unsafe { dev.cmd_copy_image_to_buffer(cmd, i.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, tmp.buffer, &region) };
            }
            all_to_host(gpu, cmd);
            let v = sub.submit(gpu, timeline, cmd, &[])?;
            timeline.wait(gpu, v, u64::MAX)?;
            let m = tmp.mapped_ref().expect("host buffer");
            let n = px as usize;
            let at = |k: usize, i: usize, size: usize| &m[offsets[k] as usize + i * size..][..size];
            Ok(Frame {
                width: w,
                height: h,
                depth: (0..n).map(|i| f32::from_le_bytes(at(0, i, 4).try_into().unwrap())).collect(),
                normal: (0..n)
                    .map(|i| {
                        let b = at(1, i, 4);
                        [i16::from_le_bytes([b[0], b[1]]), i16::from_le_bytes([b[2], b[3]])]
                    })
                    .collect(),
                material: (0..n).map(|i| u16::from_le_bytes(at(2, i, 2).try_into().unwrap())).collect(),
                surface: (0..n)
                    .map(|i| {
                        let b = at(3, i, 8);
                        [u32::from_le_bytes(b[0..4].try_into().unwrap()), u32::from_le_bytes(b[4..8].try_into().unwrap())]
                    })
                    .collect(),
            })
        })();
        gpu.wait_idle()?;
        sub.destroy(gpu);
        alloc.free(gpu, tmp);
        result
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
