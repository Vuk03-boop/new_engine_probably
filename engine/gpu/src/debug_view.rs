//! Phase 2C-2: debug views of the G-buffer (`shaders/debug_view.slang`), one fullscreen triangle
//! drawn into a colour image of the targets' size (the swapchain image in the viewer). M1 adds
//! [`View::Lit`]: simple sun and ambient shading from the same values ([`Lighting`]); no shadows.
//!
//! - The targets are read with `Load`, so every shown value is the stored value, unfiltered.
//! - Material rows come from the world's `MaterialRegistry` (base colour and emissive, indexed by
//!   `MaterialId`), uploaded once under `Category::GpuMaterial`.
//! - The region table (key and 1C snapshot id per region index) is uploaded with each set of
//!   meshes under `Category::GpuMesh`, so the snapshot-version view shows what each region was
//!   built from.
//! - The reflection of both modules is checked against [`host_layout`] before any Vulkan object is
//!   created.
//! - 2E: [`Source::Ray`] shows the ray-query outputs ([`RayTargets`]) instead of the G-buffer, with
//!   the same views; the region table follows the scene through [`Tables::replace_regions`].
//! - 3B: [`View::Light`] tone-maps the real-time lighting (`gpu::shade`, linear HDR per pixel, bound
//!   with [`DebugView::bind_radiance`]) with `Lighting::light_exposure` and the same ACES fit. The
//!   G-buffer transition is split out ([`targets_to_read`]) so a compute pass can read the targets
//!   before the view is drawn ([`DebugView::draw`]).
//! - 3D: [`View::HistoryAge`] and [`View::HistoryReason`] show the temporal state that `gpu::temporal`
//!   writes into the radiance buffer's w; [`View::Motion`] the motion buffer ([`DebugView::bind_motion`]).

use std::mem::{offset_of, size_of};

use ash::vk;
use memory::Category;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::layout::RegionKey;
use crate::mesh::GpuMeshes;
use crate::raster::{Camera, Targets};
use crate::ray::RayTargets;
use crate::reflect::{self, Field, Param, Varying};
use crate::staging::Uploader;
use crate::timeline::Timeline;
use world::MaterialParams;

pub const VS_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/debug_view_vs.spv"));
pub const VS_REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/debug_view_vs.json"));
pub const FS_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/debug_view_fs.spv"));
pub const FS_REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/debug_view_fs.json"));

/// The views, in the shader's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Material,
    Normal,
    Region,
    Brick,
    Surface,
    Depth,
    Snapshot,
    /// Sun and ambient shading (M1).
    Lit,
    /// The real-time lighting of `gpu::shade` (3B), tone-mapped (with `gpu::temporal`, the history).
    Light,
    /// 3D: history length per pixel (black 0, then blue to white at 64 samples).
    HistoryAge,
    /// 3D: the reason of the pixel's last history decision (ADR-0006 colours).
    HistoryReason,
    /// 3D: motion to the previous frame (hue: direction, brightness: length up to 16 px).
    Motion,
}

impl View {
    pub const ALL: [View; 12] = [View::Material, View::Normal, View::Region, View::Brick, View::Surface, View::Depth, View::Snapshot, View::Lit, View::Light, View::HistoryAge, View::HistoryReason, View::Motion];

    /// Whether the view shows G-buffer values (checked against the CPU re-derivation in the raster
    /// and edit tests); the others show lighting and temporal state.
    pub fn reads_gbuffer(self) -> bool {
        !matches!(self, View::Light | View::HistoryAge | View::HistoryReason | View::Motion)
    }

    pub fn name(self) -> &'static str {
        match self {
            View::Material => "material",
            View::Normal => "normal",
            View::Region => "region hash",
            View::Brick => "brick hash",
            View::Surface => "surface-id hash",
            View::Depth => "linear depth",
            View::Snapshot => "snapshot version",
            View::Lit => "lit",
            View::Light => "light",
            View::HistoryAge => "history age",
            View::HistoryReason => "history reason",
            View::Motion => "motion",
        }
    }
}

/// Which visibility result the views read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The raster G-buffer images.
    Raster,
    /// The ray-query outputs (the ray pass must have run for this frame).
    Ray,
}

/// The lit view's tunables. Sun and ambient are in the same relative units as material colours.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// Toward the sun; normalized when used.
    pub sun_dir: [f64; 3],
    pub sun_intensity: f32,
    pub ambient: f32,
    pub exposure: f32,
    /// Exposure of [`View::Light`] (ADR-0005 radiance, E_SUN = 1). Display only.
    pub light_exposure: f32,
    /// 4B: [`View::Light`] adds each surface pixel's emitted radiance (the lights are on). Off by
    /// default: the M3 street never lights.
    pub emission: bool,
}

impl Lighting {
    /// Defaults for the street block, chosen by eye from the swept renders in the M1 record.
    pub fn new(sun_dir: [f64; 3]) -> Lighting {
        Lighting { sun_dir, sun_intensity: 3.0, ambient: 0.6, exposure: 0.5, light_exposure: 16.0, emission: false }
    }
}

/// `Params::origin[3]`: the view index in bits 0-7, and this bit when [`View::Light`] adds emission.
pub const EMISSION_BIT: i32 = 256;

/// One material table row, as the shader's `MaterialRow`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialRow {
    pub base: [f32; 4],
    pub emissive: [f32; 4],
}

/// Host mirror of the shader's `Params` push constants: 128 B, the smallest `maxPushConstantsSize`
/// Vulkan guarantees.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub forward: [f32; 4],
    pub eye: [f32; 4],
    pub origin: [i32; 4],
    pub info: [u32; 4],
    pub sun: [f32; 4],
    pub light: [f32; 4],
}

const _: () = assert!(size_of::<Params>() == 128);

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

/// What the host code assumes about both modules.
pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "depth_tex", binding: 0, element: vec![] },
        Param::Descriptor { name: "normal_tex", binding: 1, element: vec![] },
        Param::Descriptor { name: "material_tex", binding: 2, element: vec![] },
        Param::Descriptor { name: "surface_tex", binding: 3, element: vec![] },
        Param::Descriptor { name: "materials", binding: 4, element: vec![field!(MaterialRow, base), field!(MaterialRow, emissive)] },
        Param::Descriptor { name: "regions", binding: 5, element: vec![] },
        Param::Descriptor { name: "ray_hits", binding: 6, element: vec![] },
        Param::Descriptor { name: "ray_surfaces", binding: 7, element: vec![] },
        Param::Descriptor { name: "radiance", binding: 8, element: vec![] },
        Param::Descriptor { name: "motion", binding: 9, element: vec![] },
        Param::PushConstants { name: "params", fields: vec![field!(Params, right), field!(Params, up), field!(Params, forward), field!(Params, eye), field!(Params, origin), field!(Params, info), field!(Params, sun), field!(Params, light)] },
    ]
}

/// Checks both modules' reflection: bindings, push constants, and the single colour output.
pub fn check_reflection(vs: &str, fs: &str) -> std::result::Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for (stage, json) in [("vertex", vs), ("fragment", fs)] {
        if let Err(e) = reflect::check(json, &host_layout()) {
            errors.extend(e.into_iter().map(|e| format!("{stage}: {e}")));
        }
    }
    let want_out = vec![Varying { name: String::new(), location: 0, scalar: "float32".into(), components: 4 }];
    match (reflect::varyings(vs), reflect::varyings(fs)) {
        (Ok((vin, _)), Ok((_, fout))) => {
            if !vin.is_empty() {
                errors.push(format!("vertex inputs {vin:?}, host expects none"));
            }
            // The fragment result is a bare float4, reflected without a name.
            if fout != want_out {
                errors.push(format!("fragment outputs {fout:?}, host expects {want_out:?}"));
            }
        }
        (a, b) => errors.push(format!("varyings: {:?} {:?}", a.err(), b.err())),
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

fn module(gpu: &Gpu, spirv: &[u8]) -> Result<vk::ShaderModule> {
    let (chunks, rest) = spirv.as_chunks::<4>();
    assert!(rest.is_empty(), "SPIR-V is whole words");
    let words: Vec<u32> = chunks.iter().map(|&c| u32::from_le_bytes(c)).collect();
    unsafe { gpu.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }.vk("vkCreateShaderModule")
}

/// Device tables the views read: material rows and the region table.
pub struct Tables {
    pub materials: Buffer,
    pub material_count: u32,
    pub regions: Buffer,
    pub region_count: u32,
}

impl Tables {
    /// Uploads one [`MaterialRow`] per material (by `MaterialId`) and one `(key, snapshot)` row per
    /// region of `meshes`, in draw (region index) order. Returns the tables and the timeline value
    /// after which they are on the device.
    pub fn upload(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, materials: &[MaterialParams], meshes: &GpuMeshes, snapshot: u64) -> Result<(Tables, u64)> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let (b, e) = (|m: &MaterialParams| m.base_color, |m: &MaterialParams| m.emissive);
        let mat_bytes: Vec<u8> = materials.iter().flat_map(|m| [b(m)[0], b(m)[1], b(m)[2], 0.0, e(m)[0], e(m)[1], e(m)[2], 0.0]).flat_map(f32::to_le_bytes).collect();
        debug_assert_eq!(mat_bytes.len(), materials.len() * size_of::<MaterialRow>());
        let rows: Vec<(RegionKey, u64)> = meshes.regions.keys().map(|&k| (k, snapshot)).collect();
        let reg_bytes = region_bytes(&rows);
        let material_count = materials.len() as u32;
        let materials = alloc.create_buffer(gpu, (mat_bytes.len() as u64).max(16), usage, Category::GpuMaterial, Kind::Device)?;
        let regions = match alloc.create_buffer(gpu, (reg_bytes.len() as u64).max(16), usage, Category::GpuMesh, Kind::Device) {
            Ok(b) => b,
            Err(e) => {
                alloc.free(gpu, materials);
                return Err(e);
            }
        };
        up.upload(gpu, timeline, &materials, 0, &mat_bytes)?;
        up.upload(gpu, timeline, &regions, 0, &reg_bytes)?;
        let v = up.flush(gpu, timeline)?;
        Ok((Tables { materials, material_count, regions, region_count: meshes.regions.len() as u32 }, v))
    }

    /// Uploads a new region table (2E: after a scene update), one `(key, snapshot)` row per region in
    /// region index order. Returns the old table buffer, which the GPU may still read (retire it),
    /// and the timeline value after which the new one is on the device. Rebind the view afterwards.
    pub fn replace_regions(&mut self, gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, rows: &[(RegionKey, u64)]) -> Result<(Buffer, u64)> {
        let bytes = region_bytes(rows);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let regions = alloc.create_buffer(gpu, (bytes.len() as u64).max(16), usage, Category::GpuMesh, Kind::Device)?;
        up.upload(gpu, timeline, &regions, 0, &bytes)?;
        let v = up.flush(gpu, timeline)?;
        self.region_count = rows.len() as u32;
        Ok((std::mem::replace(&mut self.regions, regions), v))
    }

    /// Frees now: the caller makes sure the GPU is done with them.
    pub fn free_now(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free(gpu, self.materials);
        alloc.free(gpu, self.regions);
    }
}

fn region_bytes(rows: &[(RegionKey, u64)]) -> Vec<u8> {
    rows.iter().flat_map(|(k, s)| [k.x, k.y, k.z, *s as i32]).flat_map(i32::to_le_bytes).collect()
}

pub struct DebugView {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pub format: vk::Format,
}

impl DebugView {
    /// A pipeline drawing into colour images of `format`.
    pub fn new(gpu: &Gpu, format: vk::Format) -> Result<DebugView> {
        Self::with_reflection(gpu, format, VS_REFLECTION, FS_REFLECTION)
    }

    pub fn with_reflection(gpu: &Gpu, format: vk::Format, vs_reflection: &str, fs_reflection: &str) -> Result<DebugView> {
        check_reflection(vs_reflection, fs_reflection).map_err(GpuError::Layout)?;
        let dev = &gpu.device;
        let b = |i: u32, ty| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [
            b(0, vk::DescriptorType::SAMPLED_IMAGE),
            b(1, vk::DescriptorType::SAMPLED_IMAGE),
            b(2, vk::DescriptorType::SAMPLED_IMAGE),
            b(3, vk::DescriptorType::SAMPLED_IMAGE),
            b(4, vk::DescriptorType::STORAGE_BUFFER),
            b(5, vk::DescriptorType::STORAGE_BUFFER),
            b(6, vk::DescriptorType::STORAGE_BUFFER),
            b(7, vk::DescriptorType::STORAGE_BUFFER),
            b(8, vk::DescriptorType::STORAGE_BUFFER),
            b(9, vk::DescriptorType::STORAGE_BUFFER),
        ];
        let set_layout = unsafe { dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.vk("vkCreateDescriptorSetLayout")?;
        let pc = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::FRAGMENT).offset(0).size(size_of::<Params>() as u32)];
        let sl = [set_layout];
        let layout = unsafe { dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&pc), None) }.vk("vkCreatePipelineLayout")?;
        let vs = module(gpu, VS_SPIRV)?;
        let fs = module(gpu, FS_SPIRV)?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::NONE).line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachments = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
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
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 4 }, vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 6 }];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        let set = unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sl)) }.vk("vkAllocateDescriptorSets")?[0];
        Ok(DebugView { set_layout, layout, pipeline, pool, set, format })
    }

    /// Points the descriptor set at `targets`, `tables` and the ray outputs (without them,
    /// [`Source::Ray`] must not be used; the material table stands in). The radiance and motion
    /// bindings are reset to the material table too: call [`DebugView::bind_radiance`] and
    /// [`DebugView::bind_motion`] afterwards to use the lighting and temporal views.
    /// The GPU must not be using the set (call it at start-up, or after waiting for the frames in
    /// flight, e.g. on resize or after a scene update).
    pub fn bind(&self, gpu: &Gpu, targets: &Targets, tables: &Tables, ray: Option<&RayTargets>) {
        let img = |v: vk::ImageView| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: v, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
        let images = [img(targets.depth.view), img(targets.color[0].view), img(targets.color[1].view), img(targets.color[2].view)];
        let whole = |b: vk::Buffer| [vk::DescriptorBufferInfo { buffer: b, offset: 0, range: vk::WHOLE_SIZE }];
        let (hits, surfaces) = ray.map_or((tables.materials.buffer, tables.materials.buffer), |r| (r.hits.buffer, r.surfaces.buffer));
        let bufs = [whole(tables.materials.buffer), whole(tables.regions.buffer), whole(hits), whole(surfaces), whole(tables.materials.buffer), whole(tables.materials.buffer)];
        let mut writes: Vec<_> = images.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(info)).collect();
        writes.extend(bufs.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(4 + i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)));
        unsafe { gpu.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Points the radiance binding ([`View::Light`]) at `radiance` (one RGBA32F per pixel, row-major,
    /// the targets' size). Same rule as [`DebugView::bind`]: the GPU must not be using the set.
    pub fn bind_radiance(&self, gpu: &Gpu, radiance: &Buffer) {
        let info = [vk::DescriptorBufferInfo { buffer: radiance.buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let w = [vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(8).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)];
        unsafe { gpu.device.update_descriptor_sets(&w, &[]) };
    }

    /// Points the motion binding ([`View::Motion`]) at `motion` (`gpu::temporal::History::motion`).
    /// Same rule as [`DebugView::bind`].
    pub fn bind_motion(&self, gpu: &Gpu, motion: &Buffer) {
        let info = [vk::DescriptorBufferInfo { buffer: motion.buffer, offset: 0, range: vk::WHOLE_SIZE }];
        let w = [vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(9).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)];
        unsafe { gpu.device.update_descriptor_sets(&w, &[]) };
    }

    /// Records: targets from their attachment layouts (as `Raster::record` leaves them) to
    /// shader-read, then the view of `source` drawn into `out` (already in `COLOR_ATTACHMENT_OPTIMAL`).
    /// For [`Source::Ray`] the caller has recorded the trace and a compute-write to fragment-read
    /// barrier; the raster pass still runs (the images are bound either way).
    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, gpu: &Gpu, cmd: vk::CommandBuffer, targets: &Targets, tables: &Tables, out: vk::ImageView, camera: &Camera, view: View, lighting: &Lighting, source: Source) {
        targets_to_read(gpu, cmd, targets);
        self.draw(gpu, cmd, targets, tables, out, camera, view, lighting, source);
    }

    /// Records the view only: the targets are already in `SHADER_READ_ONLY_OPTIMAL` and visible to
    /// the fragment stage ([`targets_to_read`]); for [`View::Light`] the caller has also made the
    /// radiance writes visible to it.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&self, gpu: &Gpu, cmd: vk::CommandBuffer, targets: &Targets, tables: &Tables, out: vk::ImageView, camera: &Camera, view: View, lighting: &Lighting, source: Source) {
        let dev = &gpu.device;
        let color = [vk::RenderingAttachmentInfo::default()
            .image_view(out)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: targets.extent };
        let info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&color);
        let viewport = [vk::Viewport { x: 0.0, y: 0.0, width: targets.extent.width as f32, height: targets.extent.height as f32, min_depth: 0.0, max_depth: 1.0 }];
        let mut p = params(camera, view, targets.extent, tables, lighting);
        p.light[2] = if source == Source::Ray { 1.0 } else { 0.0 };
        let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
        unsafe {
            dev.cmd_begin_rendering(cmd, &info);
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            dev.cmd_set_viewport(cmd, 0, &viewport);
            dev.cmd_set_scissor(cmd, 0, &[area]);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[self.set], &[]);
            dev.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, bytes);
            dev.cmd_draw(cmd, 3, 1, 0, 0);
            dev.cmd_end_rendering(cmd);
        }
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            gpu.device.destroy_descriptor_pool(self.pool, None);
            gpu.device.destroy_pipeline(self.pipeline, None);
            gpu.device.destroy_pipeline_layout(self.layout, None);
            gpu.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// Push constants for `camera` and `view`: the same basis and pixel-centre rule as
/// [`Camera::dir`], relative to [`Camera::origin`].
pub fn params(camera: &Camera, view: View, extent: vk::Extent2D, tables: &Tables, lighting: &Lighting) -> Params {
    let o = camera.origin();
    let s = |v: [f64; 3], k: f64| [(v[0] * k) as f32, (v[1] * k) as f32, (v[2] * k) as f32, 0.0];
    let f = [0, 1, 2].map(|a| (camera.eye[a] - o[a] as f64) as f32);
    let index = View::ALL.iter().position(|&v| v == view).expect("listed") as i32;
    Params {
        right: s(camera.right, camera.tan_half_x),
        up: s(camera.up, camera.tan_half_y),
        forward: s(camera.forward, 1.0),
        eye: [f[0], f[1], f[2], camera.near as f32],
        origin: [o[0], o[1], o[2], index | if lighting.emission && view == View::Light { EMISSION_BIT } else { 0 }],
        info: [extent.width, extent.height, tables.material_count, tables.region_count],
        sun: {
            let d = lighting.sun_dir;
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            [(d[0] / l) as f32, (d[1] / l) as f32, (d[2] / l) as f32, lighting.sun_intensity]
        },
        light: [lighting.ambient, lighting.exposure, 0.0, lighting.light_exposure],
    }
}

/// Moves the G-buffer targets from their attachment layouts (as `Raster::record` leaves them) to
/// `SHADER_READ_ONLY_OPTIMAL`, visible to the fragment and compute stages (the view and `gpu::shade`).
pub fn targets_to_read(gpu: &Gpu, cmd: vk::CommandBuffer, targets: &Targets) {
    let range = |aspect| vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
    let readers = vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER;
    let mut barriers: Vec<_> = targets
        .color
        .iter()
        .map(|i| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(readers)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(i.image)
                .subresource_range(range(vk::ImageAspectFlags::COLOR))
        })
        .collect();
    barriers.push(
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(readers)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(targets.depth.image)
            .subresource_range(range(vk::ImageAspectFlags::DEPTH)),
    );
    unsafe { gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&barriers)) };
}
