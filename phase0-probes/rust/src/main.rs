//! Phase 0 RT probe (Rust + ash). Throwaway: checks device selection, AS builds, RT pipeline creation
//! from pre-compiled SPIR-V, traced images, timestamps and memory. Not engine code.
//!
//! usage: rt_probe <shader_dir> <out.ppm> [--size W H] [--fbx scene.fbx] [--textures-out list.txt] [--textures]
//! Without --textures this is probe v2 and must stay behaviourally identical to ../cpp/main.cpp
//! (same scene, camera, math order, output). --textures (v3, Rust only, needs --fbx and the
//! shaders/out/slang_tex shader set) adds base-colour textures and alpha-tested geometry.
//! The C++-only PROBE_PICK diagnostic is not ported.

use ash::vk;
use ash::vk::Handle;
use std::ffi::{c_void, CStr};
use std::io::Write;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Instant;

static VALIDATION_ERRORS: AtomicI32 = AtomicI32::new(0);
static VALIDATION_WARNINGS: AtomicI32 = AtomicI32::new(0);

unsafe extern "system" fn debug_cb(
    sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
    } else if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        VALIDATION_WARNINGS.fetch_add(1, Ordering::Relaxed);
    }
    let msg = if (*data).p_message.is_null() {
        std::borrow::Cow::Borrowed("")
    } else {
        CStr::from_ptr((*data).p_message).to_string_lossy()
    };
    eprintln!("[validation] {msg}");
    vk::FALSE
}

fn fail(msg: &str) -> ! {
    eprintln!("FAIL {msg}");
    std::process::exit(1);
}

trait Check<T> {
    fn check(self, what: &str) -> T;
}
impl<T> Check<T> for Result<T, vk::Result> {
    fn check(self, what: &str) -> T {
        self.unwrap_or_else(|e| fail(&format!("{what} -> {e:?}")))
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) & !(a - 1)
}

// ---------------------------------------------------------------- scene
// Triangles in world space. nrm[i] = (geometric normal of triangle i, 1 for box / 0 otherwise).
#[derive(Default)]
struct Scene {
    pos: Vec<f32>,
    idx: Vec<u32>,
    nrm: Vec<f32>,
    cam: [f32; 20], // push constants: origin, forward, right (scaled), up (scaled), sun dir (w: opaque count bits)
    // --textures only: per-corner UVs, per-triangle base-colour texture index (u32::MAX = none),
    // triangles ordered [opaque..., alpha-tested...], and the FBX texture file list (index = typed_id).
    uv: Vec<f32>,
    tri_tex: Vec<u32>,
    opaque_count: u32,
    texture_files: Vec<String>,
    name: String,
    camera_name: String,
    sun_source: String,
    meshes: usize,
    nodes: usize,
    cameras: usize,
    textures: usize,
    lights: usize,
}

impl Scene {
    fn add_quad(&mut self, p: [[f32; 3]; 4], n: [f32; 3], flag: f32) {
        let base = (self.pos.len() / 3) as u32;
        for v in p {
            self.pos.extend_from_slice(&v);
        }
        self.idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        for _ in 0..2 {
            self.nrm.extend_from_slice(&[n[0], n[1], n[2], flag]);
        }
    }

    fn add_box(&mut self, x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32) {
        let faces = [
            [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], // +x
            [[x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0]], // -x
            [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]], // +y
            [[x0, y0, z1], [x0, y0, z0], [x1, y0, z0], [x1, y0, z1]], // -y
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], // +z
            [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], // -z
        ];
        let normals = [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]];
        for (f, n) in faces.into_iter().zip(normals) {
            self.add_quad(f, n, 1.0);
        }
    }

    // Direction towards the sun used when a scene has no directional light.
    fn set_default_sun(&mut self) {
        let l = (0.5f64 * 0.5 + 1.0 * 1.0 + 0.3 * 0.3).sqrt();
        self.cam[16] = (0.5 / l) as f32;
        self.cam[17] = (1.0 / l) as f32;
        self.cam[18] = (0.3 / l) as f32;
    }

    fn small(w: u32, h: u32) -> Scene {
        let mut s = Scene { name: "small".into(), sun_source: "default".into(), ..Default::default() };
        s.add_quad([[-10., 0., -10.], [-10., 0., 10.], [10., 0., 10.], [10., 0., -10.]], [0., 1., 0.], 0.0);
        s.add_box(-0.5, 0.0, -0.5, 0.5, 1.0, 0.5);
        s.add_box(1.0, 0.0, -2.0, 2.0, 2.5, -1.0);
        let aspect = w as f32 / h as f32;
        let cam = [0., 1.5, 5., 0., 0., -0.15, -1., 0., aspect * 0.6, 0., 0., 0., 0., 0.6, 0., 0.];
        s.cam[..16].copy_from_slice(&cam);
        s.set_default_sun();
        s
    }

    // Every mesh instance flattened into one world-space triangle list, meters, right-handed Y-up.
    // Camera: first FBX camera at its default pose.
    fn fbx(path: &str, w: u32, h: u32, textures_out: Option<&str>, textured: bool) -> Scene {
        let opts = ufbx::LoadOpts {
            target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
            target_camera_axes: ufbx::CoordinateAxes::right_handed_y_up(),
            target_unit_meters: 1.0,
            space_conversion: ufbx::SpaceConversion::ModifyGeometry,
            ..Default::default()
        };
        let sc = ufbx::load_file(path, opts).unwrap_or_else(|e| fail(&format!("ufbx: {}", &*e.description)));
        let mut s = Scene {
            name: "fbx".into(),
            sun_source: "default".into(),
            meshes: sc.meshes.len(),
            nodes: sc.nodes.len(),
            cameras: sc.cameras.len(),
            textures: sc.textures.len(),
            lights: sc.lights.len(),
            ..Default::default()
        };
        // Alpha-tested = the base-colour texture is BC3/DXT5 (has an alpha channel). A heuristic for
        // Bistro: foliage leaves, masked glass and string lights; BC1 trunks/walls stay opaque.
        s.texture_files = sc.textures.iter().map(|t| t.filename.to_string()).collect();
        // PROBE_NO_ALPHA is an ablation: everything opaque, same textures, to show what alpha testing changes.
        let tex_alpha: Vec<bool> = if textured && std::env::var_os("PROBE_NO_ALPHA").is_none() {
            s.texture_files.iter().map(|f| dds_fourcc(f) == *b"DXT5").collect()
        } else {
            Vec::new()
        };
        // Alpha-tested triangles are collected separately and appended after the opaque ones.
        let (mut a_idx, mut a_nrm, mut a_uv, mut a_tex) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut tri: Vec<u32> = Vec::new();
        for node in sc.nodes.iter() {
            let Some(mesh) = node.mesh.as_ref() else { continue };
            let base = (s.pos.len() / 3) as u32;
            for &v in mesh.vertices.iter() {
                let p = ufbx::transform_position(&node.geometry_to_world, v);
                s.pos.extend_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
            }
            tri.resize(mesh.max_face_triangles * 3, 0);
            for (fi, &face) in mesh.faces.iter().enumerate() {
                let nt = ufbx::triangulate_face(&mut tri, mesh, face);
                let tex = if textured {
                    mesh.face_material
                        .get(fi)
                        .and_then(|&mi| mesh.materials.get(mi as usize))
                        .and_then(|m| m.pbr.base_color.texture.as_ref().or(m.fbx.diffuse_color.texture.as_ref()))
                        .map_or(u32::MAX, |t| t.element.typed_id)
                } else {
                    u32::MAX
                };
                let alpha = tex_alpha.get(tex as usize).copied().unwrap_or(false);
                for t in 0..nt as usize {
                    let ia = base + mesh.vertex_indices[tri[t * 3] as usize];
                    let ib = base + mesh.vertex_indices[tri[t * 3 + 1] as usize];
                    let ic = base + mesh.vertex_indices[tri[t * 3 + 2] as usize];
                    let (idx, nrm, uvs, texs) = if alpha {
                        (&mut a_idx, &mut a_nrm, &mut a_uv, &mut a_tex)
                    } else {
                        (&mut s.idx, &mut s.nrm, &mut s.uv, &mut s.tri_tex)
                    };
                    idx.extend_from_slice(&[ia, ib, ic]);
                    if textured {
                        texs.push(tex);
                        for k in 0..3 {
                            // FBX UV origin is bottom-left, DDS rows start at the top: flip V.
                            let uv = if mesh.vertex_uv.exists {
                                let corner = tri[t * 3 + k] as usize;
                                mesh.vertex_uv.values[mesh.vertex_uv.indices[corner] as usize]
                            } else {
                                ufbx::Vec2 { x: 0.0, y: 1.0 }
                            };
                            uvs.extend_from_slice(&[uv.x as f32, 1.0 - uv.y as f32]);
                        }
                    }
                    let (a, b, c) = (ia as usize * 3, ib as usize * 3, ic as usize * 3);
                    let e1 = [s.pos[b] - s.pos[a], s.pos[b + 1] - s.pos[a + 1], s.pos[b + 2] - s.pos[a + 2]];
                    let e2 = [s.pos[c] - s.pos[a], s.pos[c + 1] - s.pos[a + 1], s.pos[c + 2] - s.pos[a + 2]];
                    let n = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    if len > 1e-20f32 {
                        nrm.extend_from_slice(&[n[0] / len, n[1] / len, n[2] / len, 0.0]);
                    } else {
                        nrm.extend_from_slice(&[0.0, 1.0, 0.0, 0.0]); // degenerate triangle
                    }
                }
            }
        }
        s.opaque_count = (s.idx.len() / 3) as u32;
        s.idx.extend_from_slice(&a_idx);
        s.nrm.extend_from_slice(&a_nrm);
        s.uv.extend_from_slice(&a_uv);
        s.tri_tex.extend_from_slice(&a_tex);

        let aspect = w as f64 / h as f64;
        let camera = sc.cameras.first().unwrap_or_else(|| fail("fbx has no perspective camera"));
        if camera.element.instances.is_empty() || camera.projection_mode != ufbx::ProjectionMode::Perspective {
            fail("fbx has no perspective camera");
        }
        let m = &camera.element.instances[0].node_to_world;
        let norm = |x: f64, y: f64, z: f64| {
            let l = (x * x + y * y + z * z).sqrt();
            [x / l, y / l, z / l]
        };
        let r = norm(m.m00, m.m10, m.m20);
        let u = norm(m.m01, m.m11, m.m21);
        let f = norm(m.m02, m.m12, m.m22);
        let ty = camera.field_of_view_tan.y;
        let cam = [
            m.m03 as f32,
            m.m13 as f32,
            m.m23 as f32,
            0.,
            -f[0] as f32,
            -f[1] as f32,
            -f[2] as f32,
            0.,
            (r[0] * aspect * ty) as f32,
            (r[1] * aspect * ty) as f32,
            (r[2] * aspect * ty) as f32,
            0.,
            (u[0] * ty) as f32,
            (u[1] * ty) as f32,
            (u[2] * ty) as f32,
            0.,
        ];
        s.cam[..16].copy_from_slice(&cam);
        s.camera_name = camera.element.name.to_string();

        // Sun: first directional light; ufbx gives its aim direction in node space, we need the direction towards it.
        s.set_default_sun();
        if let Some(light) = sc
            .lights
            .iter()
            .find(|l| l.type_ == ufbx::LightType::Directional && !l.element.instances.is_empty())
        {
            let d = ufbx::transform_direction(&light.element.instances[0].node_to_world, light.local_direction);
            let l = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
            s.cam[16] = (-d.x / l) as f32;
            s.cam[17] = (-d.y / l) as f32;
            s.cam[18] = (-d.z / l) as f32;
            s.sun_source = light.element.name.to_string();
        }
        s.cam[19] = f32::from_bits(s.opaque_count);

        if let Some(path) = textures_out {
            let list: String = sc.textures.iter().map(|t| format!("{}\n", &*t.filename)).collect();
            std::fs::write(path, list).unwrap_or_else(|_| fail(&format!("cannot write {path}")));
        }
        s
    }
}

// ---------------------------------------------------------------- RenderDoc
/// RenderDoc in-app API (renderdoc_app.h, RENDERDOC_API_1_6_0). Only used when the RenderDoc Vulkan
/// layer is already loaded into the process; the table is indexed by entry position in that header.
struct RenderDoc {
    table: *const *const c_void,
    _lib: libloading::os::windows::Library,
}

impl RenderDoc {
    const SET_CAPTURE_FILE_PATH_TEMPLATE: usize = 11;
    const GET_NUM_CAPTURES: usize = 13;
    const GET_CAPTURE: usize = 14;
    const START_FRAME_CAPTURE: usize = 19;
    const END_FRAME_CAPTURE: usize = 21;

    unsafe fn attach(path_template: &str) -> Option<RenderDoc> {
        let lib = libloading::os::windows::Library::open_already_loaded("renderdoc.dll").ok()?;
        let get_api: libloading::os::windows::Symbol<unsafe extern "C" fn(u32, *mut *mut c_void) -> i32> =
            lib.get(b"RENDERDOC_GetAPI\0").ok()?;
        let mut table: *mut c_void = std::ptr::null_mut();
        if get_api(10600, &mut table) != 1 || table.is_null() {
            return None;
        }
        let rd = RenderDoc { table: table as *const *const c_void, _lib: lib };
        let template = std::ffi::CString::new(path_template).ok()?;
        rd.entry::<unsafe extern "C" fn(*const std::ffi::c_char)>(Self::SET_CAPTURE_FILE_PATH_TEMPLATE)(template.as_ptr());
        Some(rd)
    }

    unsafe fn entry<F: Copy>(&self, index: usize) -> F {
        std::mem::transmute_copy(&*self.table.add(index))
    }

    unsafe fn start(&self, device: *mut c_void) {
        self.entry::<unsafe extern "C" fn(*mut c_void, *mut c_void)>(Self::START_FRAME_CAPTURE)(device, std::ptr::null_mut());
    }

    unsafe fn end(&self, device: *mut c_void) -> u32 {
        self.entry::<unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32>(Self::END_FRAME_CAPTURE)(device, std::ptr::null_mut())
    }

    unsafe fn last_capture(&self) -> Option<String> {
        let n = self.entry::<unsafe extern "C" fn() -> u32>(Self::GET_NUM_CAPTURES)();
        if n == 0 {
            return None;
        }
        let get = self.entry::<unsafe extern "C" fn(u32, *mut std::ffi::c_char, *mut u32, *mut u64) -> u32>(Self::GET_CAPTURE);
        let mut len = 0u32;
        get(n - 1, std::ptr::null_mut(), &mut len, std::ptr::null_mut());
        let mut buf = vec![0u8; len as usize + 1];
        get(n - 1, buf.as_mut_ptr() as *mut std::ffi::c_char, &mut len, std::ptr::null_mut());
        Some(CStr::from_ptr(buf.as_ptr() as *const std::ffi::c_char).to_string_lossy().into_owned())
    }
}

// ---------------------------------------------------------------- DDS
fn dds_fourcc(path: &str) -> [u8; 4] {
    let mut head = [0u8; 88];
    let ok = std::fs::File::open(path).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut head)).is_ok();
    if !ok || &head[..4] != b"DDS " {
        fail(&format!("not a readable DDS file: {path}"));
    }
    [head[84], head[85], head[86], head[87]]
}

/// A legacy-header DDS (DXT1/DXT5/ATI2) as a Vulkan BC format, mip extents and payload offsets.
struct DdsImage {
    format: vk::Format,
    width: u32,
    height: u32,
    mips: Vec<(u32, u32, u64)>, // (width, height, byte offset into data)
    data: Vec<u8>,
}

fn load_dds(path: &str) -> DdsImage {
    let file = std::fs::read(path).unwrap_or_else(|_| fail(&format!("cannot read {path}")));
    if file.len() < 128 || &file[..4] != b"DDS " {
        fail(&format!("not a DDS file: {path}"));
    }
    let rd = |o: usize| u32::from_le_bytes([file[o], file[o + 1], file[o + 2], file[o + 3]]);
    let (height, width, mip_count) = (rd(12), rd(16), rd(28).max(1));
    let name = std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let srgb = name.ends_with("_BaseColor.dds") || name.ends_with("_Emissive.dds");
    let (format, block) = match &file[84..88] {
        b"DXT1" => (if srgb { vk::Format::BC1_RGBA_SRGB_BLOCK } else { vk::Format::BC1_RGBA_UNORM_BLOCK }, 8u64),
        b"DXT5" => (if srgb { vk::Format::BC3_SRGB_BLOCK } else { vk::Format::BC3_UNORM_BLOCK }, 16),
        b"ATI2" => (vk::Format::BC5_UNORM_BLOCK, 16),
        other => fail(&format!("unsupported DDS format {:?} in {path}", String::from_utf8_lossy(other))),
    };
    let data = file[128..].to_vec();
    let mut mips = Vec::new();
    let (mut w, mut h, mut off) = (width, height, 0u64);
    for _ in 0..mip_count {
        let size = ((w as u64 + 3) / 4) * ((h as u64 + 3) / 4) * block;
        if off + size > data.len() as u64 {
            break; // truncated chain: keep the complete levels only
        }
        mips.push((w, h, off));
        off += size;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    if mips.is_empty() {
        fail(&format!("DDS has no complete mip level: {path}"));
    }
    DdsImage { format, width, height, mips, data }
}

// ---------------------------------------------------------------- vulkan helpers
struct Buffer {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    addr: vk::DeviceAddress,
    map: *mut c_void,
}

struct Ctx {
    dev: ash::Device,
    mem: vk::PhysicalDeviceMemoryProperties,
}

impl Ctx {
    fn find_mem(&self, bits: u32, want: vk::MemoryPropertyFlags) -> u32 {
        (0..self.mem.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem.memory_types[i as usize].property_flags.contains(want))
            .unwrap_or_else(|| fail("no memory type"))
    }

    unsafe fn buffer(&self, size: u64, usage: vk::BufferUsageFlags, props: vk::MemoryPropertyFlags, data: Option<&[u8]>) -> Buffer {
        let bi = vk::BufferCreateInfo::default().size(size).usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let buf = self.dev.create_buffer(&bi, None).check("vkCreateBuffer");
        let req = self.dev.get_buffer_memory_requirements(buf);
        let mut fi = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(self.find_mem(req.memory_type_bits, props))
            .push_next(&mut fi);
        let mem = self.dev.allocate_memory(&ai, None).check("vkAllocateMemory");
        self.dev.bind_buffer_memory(buf, mem, 0).check("vkBindBufferMemory");
        let mut map = std::ptr::null_mut();
        if props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            map = self.dev.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).check("vkMapMemory");
            if let Some(d) = data {
                std::ptr::copy_nonoverlapping(d.as_ptr(), map as *mut u8, d.len());
            }
        }
        let addr = self.dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));
        Buffer { buf, mem, addr, map }
    }

    unsafe fn destroy(&self, b: &Buffer) {
        self.dev.destroy_buffer(b.buf, None);
        self.dev.free_memory(b.mem, None);
    }
}

fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

// Sum of heapUsage/heapBudget over device-local heaps (VK_EXT_memory_budget), in bytes; 0 if unavailable.
unsafe fn device_local_usage(instance: &ash::Instance, phys: vk::PhysicalDevice, have_budget: bool) -> (u64, u64) {
    if !have_budget {
        return (0, 0);
    }
    let mut mb = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut mp = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut mb);
    instance.get_physical_device_memory_properties2(phys, &mut mp);
    let props = mp.memory_properties;
    let (mut usage, mut budget) = (0, 0);
    for i in 0..props.memory_heap_count as usize {
        if props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            usage += mb.heap_usage[i];
            budget += mb.heap_budget[i];
        }
    }
    (usage, budget)
}

const WARM_TRACES: u32 = 5;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: rt_probe <shader_dir> <out.ppm> [--size W H] [--fbx scene.fbx] [--textures-out list.txt] [--textures]");
        std::process::exit(2);
    }
    let (mut w, mut h) = (1280u32, 720u32);
    let mut fbx: Option<String> = None;
    let mut textures_out: Option<String> = None;
    let mut textured = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--size" if i + 2 < args.len() => {
                w = args[i + 1].parse().unwrap_or_else(|_| fail("bad width"));
                h = args[i + 2].parse().unwrap_or_else(|_| fail("bad height"));
                i += 2;
            }
            "--fbx" if i + 1 < args.len() => {
                fbx = Some(args[i + 1].clone());
                i += 1;
            }
            "--textures-out" if i + 1 < args.len() => {
                textures_out = Some(args[i + 1].clone());
                i += 1;
            }
            "--textures" => textured = true,
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let t_start = Instant::now();
    if textured && fbx.is_none() {
        fail("--textures needs --fbx");
    }
    let scene = match &fbx {
        Some(p) => Scene::fbx(p, w, h, textures_out.as_deref(), textured),
        None => Scene::small(w, h),
    };
    let t_scene = Instant::now();
    let code = unsafe { run(&args[1], &args[2], w, h, &scene, textured, t_start, t_scene) };
    std::process::exit(code);
}

#[allow(clippy::too_many_arguments)]
unsafe fn run(
    shader_dir: &str,
    out_path: &str,
    w: u32,
    h: u32,
    scene: &Scene,
    textured: bool,
    t_start: Instant,
    t_scene: Instant,
) -> i32 {
    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
    let tri_count = (scene.idx.len() / 3) as u32;
    let vert_count = (scene.pos.len() / 3) as u32;
    let entry = ash::Entry::load().unwrap_or_else(|e| fail(&format!("load vulkan-1.dll: {e}")));

    // ---- instance (+ validation if present)
    let validation_name = c"VK_LAYER_KHRONOS_validation";
    let have_validation = std::env::var_os("PROBE_NO_VALIDATION").is_none()
        && entry
            .enumerate_instance_layer_properties()
            .check("enumerate layers")
            .iter()
            .any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == validation_name);
    let layer_ptrs = [validation_name.as_ptr()];
    let ext_ptrs = [ash::ext::debug_utils::NAME.as_ptr()];
    let app = vk::ApplicationInfo::default().application_name(c"rt_probe_rust").api_version(vk::API_VERSION_1_3);
    let dbg_info = || {
        vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_cb))
    };
    let mut dbg = dbg_info();
    let mut ici = vk::InstanceCreateInfo::default().application_info(&app);
    if have_validation {
        ici = ici.enabled_layer_names(&layer_ptrs).enabled_extension_names(&ext_ptrs).push_next(&mut dbg);
    }
    let instance = entry.create_instance(&ici, None).check("vkCreateInstance");
    let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
    let messenger = if have_validation {
        debug_utils.create_debug_utils_messenger(&dbg_info(), None).check("create messenger")
    } else {
        vk::DebugUtilsMessengerEXT::null()
    };

    // ---- pick device by capability, never by index
    let required_exts = [
        ash::khr::acceleration_structure::NAME,
        ash::khr::ray_tracing_pipeline::NAME,
        ash::khr::deferred_host_operations::NAME,
    ];
    let mut phys = vk::PhysicalDevice::null();
    let mut best_score = -1;
    let mut have_budget = false;
    for p in instance.enumerate_physical_devices().check("enumerate devices") {
        let exts = instance.enumerate_device_extension_properties(p).check("enumerate device exts");
        let has = |want: &CStr| exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == want);
        if !required_exts.iter().all(|e| has(e)) {
            continue;
        }
        let pr = instance.get_physical_device_properties(p);
        let score = if pr.device_type == vk::PhysicalDeviceType::DISCRETE_GPU { 2 } else { 1 };
        if score > best_score {
            best_score = score;
            phys = p;
            have_budget = has(ash::ext::memory_budget::NAME);
        }
    }
    if phys == vk::PhysicalDevice::null() {
        fail("no ray-tracing capable device");
    }

    let mut rtp = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut asp = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rtp).push_next(&mut asp);
    instance.get_physical_device_properties2(phys, &mut props2);
    let pr = props2.properties;
    let mem = instance.get_physical_device_memory_properties(phys);
    if rtp.max_ray_recursion_depth < 2 {
        fail("maxRayRecursionDepth < 2");
    }

    let qfi = instance
        .get_physical_device_queue_family_properties(phys)
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE) && q.timestamp_valid_bits > 0)
        .unwrap_or_else(|| fail("no compute queue with timestamps")) as u32;

    // ---- device
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .runtime_descriptor_array(textured)
        .shader_sampled_image_array_non_uniform_indexing(textured);
    let mut fas = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut frt = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
    let prio = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfi).queue_priorities(&prio)];
    let mut dev_ext_ptrs: Vec<_> = required_exts.iter().map(|e| e.as_ptr()).collect();
    if have_budget {
        dev_ext_ptrs.push(ash::ext::memory_budget::NAME.as_ptr());
    }
    let dci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&qci)
        .enabled_extension_names(&dev_ext_ptrs)
        .push_next(&mut f12)
        .push_next(&mut fas)
        .push_next(&mut frt);
    let dev = instance.create_device(phys, &dci, None).check("vkCreateDevice");
    let queue = dev.get_device_queue(qfi, 0);
    let t_device = Instant::now();
    let (vram0, _) = device_local_usage(&instance, phys, have_budget);
    let as_ext = ash::khr::acceleration_structure::Device::new(&instance, &dev);
    let rt_ext = ash::khr::ray_tracing_pipeline::Device::new(&instance, &dev);
    let c = Ctx { dev: dev.clone(), mem };

    let pool = dev
        .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(qfi), None)
        .check("vkCreateCommandPool");
    // timestamps: [0,1] AS build, then (start, end) for 1 cold + WARM_TRACES warm traces
    let n_ts = 2 + 2 * (1 + WARM_TRACES);
    let qpool = dev
        .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(n_ts), None)
        .check("vkCreateQueryPool");
    let cqpool = dev
        .create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
                .query_count(1),
            None,
        )
        .check("vkCreateQueryPool(compacted)");

    let run_cmds = |rec: &dyn Fn(vk::CommandBuffer)| {
        let cai = vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1);
        let cb = dev.allocate_command_buffers(&cai).check("vkAllocateCommandBuffers")[0];
        dev.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
            .check("vkBeginCommandBuffer");
        rec(cb);
        dev.end_command_buffer(cb).check("vkEndCommandBuffer");
        let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).check("vkCreateFence");
        let cbs = [cb];
        dev.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence).check("vkQueueSubmit");
        dev.wait_for_fences(&[fence], true, u64::MAX).check("vkWaitForFences");
        dev.destroy_fence(fence, None);
        dev.free_command_buffers(pool, &cbs);
    };

    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
    // Device-local buffer filled through a temporary host-visible staging buffer.
    let upload = |data: &[u8], usage: vk::BufferUsageFlags| {
        let size = data.len() as u64;
        let staging = c.buffer(size, vk::BufferUsageFlags::TRANSFER_SRC, host, Some(data));
        let dst = c.buffer(size, usage | vk::BufferUsageFlags::TRANSFER_DST, local, None);
        run_cmds(&|cb| dev.cmd_copy_buffer(cb, staging.buf, dst.buf, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]));
        c.destroy(&staging);
        dst
    };

    // ---- geometry upload
    let t_up0 = Instant::now();
    let as_input = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
    let vbuf = upload(bytes(&scene.pos), as_input);
    let ibuf = upload(bytes(&scene.idx), as_input);
    let nbuf = upload(bytes(&scene.nrm), vk::BufferUsageFlags::STORAGE_BUFFER);
    let (uv_buf, tex_buf) = if textured {
        (
            Some(upload(bytes(&scene.uv), vk::BufferUsageFlags::STORAGE_BUFFER)),
            Some(upload(bytes(&scene.tri_tex), vk::BufferUsageFlags::STORAGE_BUFFER)),
        )
    } else {
        (None, None)
    };
    let t_up1 = Instant::now();

    // ---- textures (--textures): every FBX texture with its full BC mip chain, sampled in ray tracing stages
    let mut tex_images: Vec<(vk::Image, vk::DeviceMemory, vk::ImageView)> = Vec::new();
    let (mut tex_payload, mut tex_alloc) = (0u64, 0u64);
    for path in scene.texture_files.iter().filter(|_| textured) {
        let dds = load_dds(path);
        let levels = dds.mips.len() as u32;
        let image = dev
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(dds.format)
                    .extent(vk::Extent3D { width: dds.width, height: dds.height, depth: 1 })
                    .mip_levels(levels)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
                None,
            )
            .check("vkCreateImage(texture)");
        let req = dev.get_image_memory_requirements(image);
        let mem = dev
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(c.find_mem(req.memory_type_bits, local)),
                None,
            )
            .check("vkAllocateMemory(texture)");
        dev.bind_image_memory(image, mem, 0).check("vkBindImageMemory(texture)");
        let staging = c.buffer(dds.data.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, host, Some(&dds.data));
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(levels)
            .layer_count(1);
        let regions: Vec<_> = dds
            .mips
            .iter()
            .enumerate()
            .map(|(level, &(mw, mh, off))| {
                vk::BufferImageCopy::default()
                    .buffer_offset(off)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(level as u32)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D { width: mw, height: mh, depth: 1 })
            })
            .collect();
        run_cmds(&|cb| {
            let to_dst = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range);
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            dev.cmd_copy_buffer_to_image(cb, staging.buf, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);
            let to_read = to_dst
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
        });
        c.destroy(&staging);
        let view = dev
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(dds.format)
                    .subresource_range(range),
                None,
            )
            .check("vkCreateImageView(texture)");
        tex_payload += dds.data.len() as u64;
        tex_alloc += req.size;
        tex_images.push((image, mem, view));
    }
    let sampler = if textured {
        dev.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )
        .check("vkCreateSampler")
    } else {
        vk::Sampler::null()
    };
    let t_tex1 = Instant::now();

    // ---- AS sizing/creation helper
    let scratch_align = asp.min_acceleration_structure_scratch_offset_alignment as u64;
    let create_as = |ty: vk::AccelerationStructureTypeKHR,
                     flags: vk::BuildAccelerationStructureFlagsKHR,
                     geos: &[vk::AccelerationStructureGeometryKHR],
                     prims: &[u32]| {
        let bgi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(geos);
        let mut sz = vk::AccelerationStructureBuildSizesInfoKHR::default();
        as_ext.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &bgi, prims, &mut sz);
        let as_buf = c.buffer(sz.acceleration_structure_size, vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR, local, None);
        let aci = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(as_buf.buf)
            .size(sz.acceleration_structure_size)
            .ty(ty);
        let accel = as_ext.create_acceleration_structure(&aci, None).check("vkCreateAccelerationStructureKHR");
        let scratch = c.buffer(sz.build_scratch_size + scratch_align, vk::BufferUsageFlags::STORAGE_BUFFER, local, None);
        (accel, as_buf, scratch, sz)
    };
    fn build_info<'a>(
        ty: vk::AccelerationStructureTypeKHR,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        g: &'a [vk::AccelerationStructureGeometryKHR<'a>],
        dst: vk::AccelerationStructureKHR,
        scratch: &Buffer,
        scratch_align: u64,
    ) -> vk::AccelerationStructureBuildGeometryInfoKHR<'a> {
        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(g)
            .dst_acceleration_structure(dst)
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: align_up(scratch.addr, scratch_align) })
    }

    // ---- BLAS
    let blas_flags =
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION;
    let tlas_flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
    let opaque_tris = if textured { scene.opaque_count } else { tri_count };
    let alpha_tris = tri_count - opaque_tris;
    let tri_data = |first_tri: u32| {
        vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vbuf.addr })
            .vertex_stride(12)
            .max_vertex(vert_count - 1)
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR { device_address: ibuf.addr + first_tri as u64 * 12 })
    };
    let mut geos = vec![vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: tri_data(0) })];
    let mut prims = vec![opaque_tris];
    if alpha_tris > 0 {
        // No OPAQUE flag: any-hit shaders run and alpha-test; each candidate is tested once.
        geos.push(
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: tri_data(opaque_tris) }),
        );
        prims.push(alpha_tris);
    }
    let (blas, blas_buf, blas_scratch, blas_sz) = create_as(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, blas_flags, &geos, &prims);
    let blas_addr = as_ext.get_acceleration_structure_device_address(
        &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blas),
    );

    // ---- TLAS
    let inst = vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR { matrix: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.] },
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            0,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: blas_addr },
    };
    let inst_bytes = std::slice::from_raw_parts(
        &inst as *const _ as *const u8,
        std::mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
    );
    let inst_buf = upload(inst_bytes, as_input);
    let tgeo = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .data(vk::DeviceOrHostAddressConstKHR { device_address: inst_buf.addr }),
        });
    let tgeos = [tgeo];
    let (tlas, tlas_buf, tlas_scratch, tlas_sz) = create_as(vk::AccelerationStructureTypeKHR::TOP_LEVEL, tlas_flags, &tgeos, &[1]);

    let blas_info = build_info(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, blas_flags, &geos, blas, &blas_scratch, scratch_align);
    let tlas_info = build_info(vk::AccelerationStructureTypeKHR::TOP_LEVEL, tlas_flags, &tgeos, tlas, &tlas_scratch, scratch_align);
    let blas_range: Vec<_> = prims.iter().map(|&p| vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(p)).collect();
    let tlas_range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(1)];
    run_cmds(&|cb| {
        dev.cmd_reset_query_pool(cb, qpool, 0, n_ts);
        dev.cmd_reset_query_pool(cb, cqpool, 0, 1);
        dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qpool, 0);
        as_ext.cmd_build_acceleration_structures(cb, std::slice::from_ref(&blas_info), &[&blas_range]);
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
        dev.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        );
        as_ext.cmd_write_acceleration_structures_properties(
            cb,
            &[blas],
            vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            cqpool,
            0,
        );
        as_ext.cmd_build_acceleration_structures(cb, std::slice::from_ref(&tlas_info), &[&tlas_range]);
        dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qpool, 1);
    });
    let mut blas_compacted = [0u64; 1];
    dev.get_query_pool_results(cqpool, 0, &mut blas_compacted, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
        .check("vkGetQueryPoolResults(compacted)");
    // Scratch is only needed during the build; free it before measuring resident memory.
    c.destroy(&blas_scratch);
    c.destroy(&tlas_scratch);
    let t_as = Instant::now();

    // ---- output image + readback
    let imci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D { width: w, height: h, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC);
    let image = dev.create_image(&imci, None).check("vkCreateImage");
    let ireq = dev.get_image_memory_requirements(image);
    let image_mem = dev
        .allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(ireq.size)
                .memory_type_index(c.find_mem(ireq.memory_type_bits, local)),
            None,
        )
        .check("vkAllocateMemory(image)");
    dev.bind_image_memory(image, image_mem, 0).check("vkBindImageMemory");
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let view = dev
        .create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(range),
            None,
        )
        .check("vkCreateImageView");
    let readback = c.buffer(
        (w * h * 4) as u64,
        vk::BufferUsageFlags::TRANSFER_DST,
        host | vk::MemoryPropertyFlags::HOST_CACHED,
        None,
    );

    // ---- descriptors
    let hit_stages = vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::ANY_HIT_KHR;
    let n_tex = tex_images.len() as u32;
    let mut binds = vec![
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
    ];
    if textured {
        let b = |i: u32, ty: vk::DescriptorType, n: u32| {
            vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(ty).descriptor_count(n).stage_flags(hit_stages)
        };
        binds.push(b(3, vk::DescriptorType::STORAGE_BUFFER, 1));
        binds.push(b(4, vk::DescriptorType::STORAGE_BUFFER, 1));
        binds.push(b(5, vk::DescriptorType::SAMPLER, 1));
        binds.push(b(6, vk::DescriptorType::SAMPLED_IMAGE, n_tex));
    }
    let dsl = dev
        .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds), None)
        .check("vkCreateDescriptorSetLayout");
    let mut psz = vec![
        vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 1 },
        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 },
        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: if textured { 3 } else { 1 } },
    ];
    if textured {
        psz.push(vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 1 });
        psz.push(vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: n_tex });
    }
    let dpool = dev
        .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&psz), None)
        .check("vkCreateDescriptorPool");
    let dsls = [dsl];
    let dset = dev
        .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(dpool).set_layouts(&dsls))
        .check("vkAllocateDescriptorSets")[0];
    let tlases = [tlas];
    let mut was = vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&tlases);
    let dii = [vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL)];
    let dbi = [vk::DescriptorBufferInfo::default().buffer(nbuf.buf).range(vk::WHOLE_SIZE)];
    let uv_info = uv_buf.as_ref().map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buf).range(vk::WHOLE_SIZE)]);
    let tex_info = tex_buf.as_ref().map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buf).range(vk::WHOLE_SIZE)]);
    let sampler_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
    let image_infos: Vec<_> = tex_images
        .iter()
        .map(|&(_, _, v)| vk::DescriptorImageInfo::default().image_view(v).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL))
        .collect();
    let mut writes = vec![
        vk::WriteDescriptorSet::default()
            .dst_set(dset)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .push_next(&mut was),
        vk::WriteDescriptorSet::default()
            .dst_set(dset)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&dii),
        vk::WriteDescriptorSet::default()
            .dst_set(dset)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&dbi),
    ];
    if let (Some(uv_info), Some(tex_info)) = (&uv_info, &tex_info) {
        let wd = |i: u32, ty: vk::DescriptorType| vk::WriteDescriptorSet::default().dst_set(dset).dst_binding(i).descriptor_type(ty);
        writes.push(wd(3, vk::DescriptorType::STORAGE_BUFFER).buffer_info(uv_info));
        writes.push(wd(4, vk::DescriptorType::STORAGE_BUFFER).buffer_info(tex_info));
        writes.push(wd(5, vk::DescriptorType::SAMPLER).image_info(&sampler_info));
        writes.push(wd(6, vk::DescriptorType::SAMPLED_IMAGE).image_info(&image_infos));
    }
    dev.update_descriptor_sets(&writes, &[]);

    // ---- pipeline
    let t_pipe0 = Instant::now();
    let mut files = vec!["rgen.spv", "miss.spv", "shadow.spv", "chit.spv"];
    let mut stages = vec![
        vk::ShaderStageFlags::RAYGEN_KHR,
        vk::ShaderStageFlags::MISS_KHR,
        vk::ShaderStageFlags::MISS_KHR,
        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
    ];
    if textured {
        files.extend(["ahit.spv", "ahit_shadow.spv"]);
        stages.extend([vk::ShaderStageFlags::ANY_HIT_KHR, vk::ShaderStageFlags::ANY_HIT_KHR]);
    }
    let modules: Vec<vk::ShaderModule> = files
        .iter()
        .map(|f| {
            let path = format!("{shader_dir}/{f}");
            let mut file = std::fs::File::open(&path).unwrap_or_else(|_| fail(&format!("cannot open {path}")));
            let words = ash::util::read_spv(&mut file).unwrap_or_else(|_| fail(&format!("bad spv {path}")));
            dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .check("vkCreateShaderModule")
        })
        .collect();
    let ssci: Vec<_> = modules
        .iter()
        .zip(stages)
        .map(|(&m, s)| vk::PipelineShaderStageCreateInfo::default().stage(s).module(m).name(c"main"))
        .collect();
    // Groups: 0 raygen, 1 miss, 2 shadow miss, 3 primary hit (chit [+ ahit]), 4 shadow hit (ahit only, textured).
    let general = |i: u32| {
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(i)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
    };
    let hit = |chit: u32, ahit: u32| {
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(chit)
            .any_hit_shader(ahit)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
    };
    let mut groups = vec![general(0), general(1), general(2)];
    if textured {
        groups.push(hit(3, 4));
        groups.push(hit(vk::SHADER_UNUSED_KHR, 5));
    } else {
        groups.push(hit(3, vk::SHADER_UNUSED_KHR));
    }
    let n_groups = groups.len();
    let n_hit = n_groups as u64 - 3;
    let pc_stages = vk::ShaderStageFlags::RAYGEN_KHR
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
        | if textured { vk::ShaderStageFlags::ANY_HIT_KHR } else { vk::ShaderStageFlags::empty() };
    let pcr = [vk::PushConstantRange { stage_flags: pc_stages, offset: 0, size: std::mem::size_of_val(&scene.cam) as u32 }];
    let layout = dev
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None)
        .check("vkCreatePipelineLayout");
    let rpci = vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(&ssci)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(2)
        .layout(layout);
    let pipeline = rt_ext
        .create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), vk::PipelineCache::null(), &[rpci], None)
        .map_err(|(_, e)| e)
        .check("vkCreateRayTracingPipelinesKHR")[0];
    let t_pipe1 = Instant::now();

    // ---- shader binding table: [raygen][miss, shadow miss][hit], each region base-aligned
    let hsize = rtp.shader_group_handle_size as usize;
    let hstride = align_up(hsize as u64, rtp.shader_group_handle_alignment as u64);
    let base = rtp.shader_group_base_alignment as u64;
    let (rgen_size, miss_size, hit_size) = (align_up(hstride, base), align_up(2 * hstride, base), align_up(n_hit * hstride, base));
    let handles = rt_ext
        .get_ray_tracing_shader_group_handles(pipeline, 0, n_groups as u32, n_groups * hsize)
        .check("vkGetRayTracingShaderGroupHandlesKHR");
    let sbt = c.buffer(rgen_size + miss_size + hit_size + base, vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR, host, None);
    let sbt_base = align_up(sbt.addr, base);
    let sbt_ptr = (sbt.map as *mut u8).add((sbt_base - sbt.addr) as usize);
    let put = |off: u64, group: usize| {
        std::ptr::copy_nonoverlapping(handles[group * hsize..].as_ptr(), sbt_ptr.add(off as usize), hsize)
    };
    put(0, 0);
    put(rgen_size, 1);
    put(rgen_size + hstride, 2);
    for k in 0..n_hit {
        put(rgen_size + miss_size + k * hstride, 3 + k as usize);
    }
    let r_rgen = vk::StridedDeviceAddressRegionKHR { device_address: sbt_base, stride: rgen_size, size: rgen_size };
    let r_miss = vk::StridedDeviceAddressRegionKHR { device_address: sbt_base + rgen_size, stride: hstride, size: miss_size };
    let r_hit = vk::StridedDeviceAddressRegionKHR {
        device_address: sbt_base + rgen_size + miss_size,
        stride: hstride,
        size: hit_size,
    };
    let r_call = vk::StridedDeviceAddressRegionKHR::default();

    let (vram1, budget1) = device_local_usage(&instance, phys, have_budget);

    // PROBE_RENDERDOC_CAPTURE=<path template>: capture the trace submission with RenderDoc. The Vulkan
    // device pointer RenderDoc expects is the instance's dispatch pointer (RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE).
    let renderdoc = std::env::var("PROBE_RENDERDOC_CAPTURE").ok().map(|template| {
        RenderDoc::attach(&template).unwrap_or_else(|| fail("PROBE_RENDERDOC_CAPTURE is set but the RenderDoc layer is not loaded"))
    });
    let rd_device = *(instance.handle().as_raw() as usize as *const *mut c_void);
    if let Some(rd) = &renderdoc {
        rd.start(rd_device);
    }

    // ---- trace (1 cold + WARM_TRACES warm, identical output) + readback
    run_cmds(&|cb| {
        let ib = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        dev.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[ib],
        );
        dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline);
        dev.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::RAY_TRACING_KHR, layout, 0, &[dset], &[]);
        dev.cmd_push_constants(cb, layout, pc_stages, 0, bytes(&scene.cam));
        for k in 0..1 + WARM_TRACES {
            if k > 0 {
                let mb = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE);
                dev.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[mb],
                    &[],
                    &[],
                );
            }
            dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qpool, 2 + 2 * k);
            rt_ext.cmd_trace_rays(cb, &r_rgen, &r_miss, &r_hit, &r_call, w, h, 1);
            dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qpool, 3 + 2 * k);
        }
        let ib2 = ib
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        dev.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[ib2],
        );
        let copy = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
        dev.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, readback.buf, &[copy]);
        let hb = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(readback.buf)
            .size(vk::WHOLE_SIZE);
        dev.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[hb],
            &[],
        );
    });

    if let Some(rd) = &renderdoc {
        if rd.end(rd_device) != 1 {
            fail("RenderDoc EndFrameCapture reported failure");
        }
        eprintln!("[renderdoc] capture: {}", rd.last_capture().unwrap_or_else(|| fail("RenderDoc wrote no capture")));
    }
    let mut ts = vec![0u64; n_ts as usize];
    dev.get_query_pool_results(qpool, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
        .check("vkGetQueryPoolResults");
    let tick_ms = pr.limits.timestamp_period as f64 / 1e6;
    let mut warm: Vec<f64> = (1..=WARM_TRACES as usize).map(|k| (ts[3 + 2 * k] - ts[2 + 2 * k]) as f64 * tick_ms).collect();
    warm.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // ---- write PPM (binary P6, RGB)
    {
        let px = std::slice::from_raw_parts(readback.map as *const u8, (w * h * 4) as usize);
        let mut out = Vec::with_capacity((w * h * 3) as usize + 32);
        write!(out, "P6\n{w} {h}\n255\n").unwrap();
        for p in px.chunks_exact(4) {
            out.extend_from_slice(&p[..3]);
        }
        std::fs::write(out_path, out).unwrap_or_else(|_| fail(&format!("cannot write {out_path}")));
    }
    let t_end = Instant::now();

    // PROBE_HOLD_MS keeps the device alive after the work so an external profiler (Nsight GPU Trace)
    // can finish its trace window; Nsight drops the report if the process exits mid-trace. Not timed.
    if let Some(ms) = std::env::var("PROBE_HOLD_MS").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }

    // ---- teardown (validation reports leaks, so destroy everything)
    // PROBE_FAULT_LEAK is a negative control: leaking the pipeline must produce validation errors.
    if std::env::var_os("PROBE_FAULT_LEAK").is_none() {
        dev.destroy_pipeline(pipeline, None);
    }
    dev.destroy_pipeline_layout(layout, None);
    for m in modules {
        dev.destroy_shader_module(m, None);
    }
    dev.destroy_descriptor_pool(dpool, None);
    dev.destroy_descriptor_set_layout(dsl, None);
    dev.destroy_image_view(view, None);
    dev.destroy_image(image, None);
    dev.free_memory(image_mem, None);
    as_ext.destroy_acceleration_structure(tlas, None);
    as_ext.destroy_acceleration_structure(blas, None);
    for &(image, mem, view) in &tex_images {
        dev.destroy_image_view(view, None);
        dev.destroy_image(image, None);
        dev.free_memory(mem, None);
    }
    dev.destroy_sampler(sampler, None);
    for b in uv_buf.iter().chain(tex_buf.iter()) {
        c.destroy(b);
    }
    for b in [&vbuf, &ibuf, &nbuf, &blas_buf, &inst_buf, &tlas_buf, &readback, &sbt] {
        c.destroy(b);
    }
    dev.destroy_query_pool(cqpool, None);
    dev.destroy_query_pool(qpool, None);
    dev.destroy_command_pool(pool, None);
    dev.destroy_device(None);
    if have_validation {
        debug_utils.destroy_debug_utils_messenger(messenger, None);
    }
    instance.destroy_instance(None);

    let mib = 1.0 / (1024.0 * 1024.0);
    let name = CStr::from_ptr(pr.device_name.as_ptr()).to_string_lossy();
    let errors = VALIDATION_ERRORS.load(Ordering::Relaxed);
    println!(
        "{{\"probe\":\"rust\",\"textured\":{textured},\"opaque_tris\":{opaque_tris},\"alpha_tris\":{alpha_tris},\
\"textures_uploaded\":{},\"texture_payload_mib\":{:.1},\"texture_alloc_mib\":{:.1},\"host_texture_ms\":{:.1},\"scene\":\"{}\",\"camera\":\"{}\",\"sun\":\"{}\",\"sun_dir\":[{:.4},{:.4},{:.4}],\"fbx_lights\":{},\
\"device\":\"{name}\",\"driver_api\":\"{}.{}.{}\",\"shader_dir\":\"{shader_dir}\",\"width\":{w},\"height\":{h},\
\"triangles\":{tri_count},\"vertices\":{vert_count},\"fbx_meshes\":{},\"fbx_nodes\":{},\"fbx_cameras\":{},\"fbx_textures\":{},\
\"validation\":{have_validation},\"validation_errors\":{errors},\"validation_warnings\":{},\"geometry_mib\":{:.2},\
\"blas_mib\":{:.2},\"blas_compacted_mib\":{:.2},\"blas_scratch_mib\":{:.2},\"tlas_bytes\":{},\"vram_before_mib\":{:.1},\
\"vram_resident_mib\":{:.1},\"vram_budget_mib\":{:.1},\"host_scene_ms\":{:.1},\"host_startup_to_device_ms\":{:.1},\
\"host_upload_ms\":{:.1},\"host_as_total_ms\":{:.1},\"host_pipeline_create_ms\":{:.3},\"host_total_ms\":{:.1},\
\"gpu_as_build_ms\":{:.3},\"gpu_trace_cold_ms\":{:.3},\"gpu_trace_warm_median_ms\":{:.3},\"gpu_trace_warm_min_ms\":{:.3},\
\"gpu_trace_warm_max_ms\":{:.3}}}",
        tex_images.len(),
        tex_payload as f64 * mib,
        tex_alloc as f64 * mib,
        ms(t_up1, t_tex1),
        scene.name,
        scene.camera_name,
        scene.sun_source,
        scene.cam[16],
        scene.cam[17],
        scene.cam[18],
        scene.lights,
        vk::api_version_major(pr.api_version),
        vk::api_version_minor(pr.api_version),
        vk::api_version_patch(pr.api_version),
        scene.meshes,
        scene.nodes,
        scene.cameras,
        scene.textures,
        VALIDATION_WARNINGS.load(Ordering::Relaxed),
        (scene.pos.len() + scene.idx.len() + scene.nrm.len()) as f64 * 4.0 * mib,
        blas_sz.acceleration_structure_size as f64 * mib,
        blas_compacted[0] as f64 * mib,
        blas_sz.build_scratch_size as f64 * mib,
        tlas_sz.acceleration_structure_size,
        vram0 as f64 * mib,
        vram1 as f64 * mib,
        budget1 as f64 * mib,
        ms(t_start, t_scene),
        ms(t_scene, t_device),
        ms(t_up0, t_up1),
        ms(t_up0, t_as),
        ms(t_pipe0, t_pipe1),
        ms(t_start, t_end),
        (ts[1] - ts[0]) as f64 * tick_ms,
        (ts[3] - ts[2]) as f64 * tick_ms,
        warm[warm.len() / 2],
        warm[0],
        warm[warm.len() - 1],
    );
    if errors > 0 {
        3
    } else {
        0
    }
}
