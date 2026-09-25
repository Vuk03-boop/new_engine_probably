//! Phase 0 window probe: does the main event loop stay responsive while an RT pipeline compiles cold?
//! Throwaway, not engine code.
//!
//! usage: window_probe <shader_dir> [--compile main|background] [--seed unique|fixed] [--delay-ms N] [--after-ms N]
//!
//! A window presents an animated clear colour every loop iteration (FIFO). After `delay-ms` of
//! baseline frames, the RT pipeline from `shader_dir` is created either inline on the main thread
//! (control arm: must show a stall) or on a background thread. `--seed unique` passes a new
//! specialization constant each run so the driver's pipeline cache cannot serve the compile.
//! Prints one JSON line with frame-interval statistics per phase.

use ash::vk;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowId};

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
    if !(*data).p_message.is_null() {
        eprintln!("[validation] {}", CStr::from_ptr((*data).p_message).to_string_lossy());
    }
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

#[derive(Clone, Copy, PartialEq)]
enum CompileOn {
    Main,
    Background,
}

struct Config {
    shader_dir: String,
    compile_on: CompileOn,
    unique_seed: bool,
    delay: Duration,
    after: Duration,
}

/// Everything the compile needs; plain handles and loaders, so it can move to another thread.
#[derive(Clone)]
struct CompileJob {
    rt: ash::khr::ray_tracing_pipeline::Device,
    modules: [vk::ShaderModule; 4],
    layout: vk::PipelineLayout,
    seed: u32,
    handle_size: u32,
}

struct CompileResult {
    pipeline: vk::Pipeline,
    start: Instant,
    end: Instant,
    handles_ok: bool,
}

fn compile(job: &CompileJob) -> CompileResult {
    let start = Instant::now();
    let seed = [job.seed];
    let entries = [vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 }];
    let spec = vk::SpecializationInfo::default().map_entries(&entries).data(unsafe {
        std::slice::from_raw_parts(seed.as_ptr() as *const u8, 4)
    });
    let stages = [
        vk::ShaderStageFlags::RAYGEN_KHR,
        vk::ShaderStageFlags::MISS_KHR,
        vk::ShaderStageFlags::MISS_KHR,
        vk::ShaderStageFlags::CLOSEST_HIT_KHR,
    ];
    let ssci: Vec<_> = (0..4)
        .map(|i| {
            let s = vk::PipelineShaderStageCreateInfo::default().stage(stages[i]).module(job.modules[i]).name(c"main");
            if i == 0 {
                s.specialization_info(&spec)
            } else {
                s
            }
        })
        .collect();
    let groups: Vec<_> = (0..4u32)
        .map(|i| {
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(if i == 3 {
                    vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
                } else {
                    vk::RayTracingShaderGroupTypeKHR::GENERAL
                })
                .general_shader(if i == 3 { vk::SHADER_UNUSED_KHR } else { i })
                .closest_hit_shader(if i == 3 { 3 } else { vk::SHADER_UNUSED_KHR })
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
        })
        .collect();
    let info = vk::RayTracingPipelineCreateInfoKHR::default()
        .stages(&ssci)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(2)
        .layout(job.layout);
    let pipeline = unsafe {
        job.rt
            .create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), vk::PipelineCache::null(), &[info], None)
            .map_err(|(_, e)| e)
            .check("vkCreateRayTracingPipelinesKHR")[0]
    };
    let end = Instant::now();
    // The pipeline must be usable, not just created: fetch its shader group handles.
    let handles_ok = unsafe {
        job.rt
            .get_ray_tracing_shader_group_handles(pipeline, 0, 4, 4 * job.handle_size as usize)
            .map(|h| h.iter().any(|&b| b != 0))
            .unwrap_or(false)
    };
    CompileResult { pipeline, start, end, handles_ok }
}

struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
    surface_ext: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    dev: ash::Device,
    swapchain_ext: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
    rendered: Vec<vk::Semaphore>,
    job: CompileJob,
    dsl: vk::DescriptorSetLayout,
    device_name: String,
    present_mode: vk::PresentModeKHR,
}

unsafe fn init_gpu(window: &Window, cfg: &Config) -> Gpu {
    let entry = ash::Entry::load().unwrap_or_else(|e| fail(&format!("load vulkan-1.dll: {e}")));
    let display = window.display_handle().unwrap_or_else(|_| fail("display handle")).as_raw();
    let whandle = window.window_handle().unwrap_or_else(|_| fail("window handle")).as_raw();

    let validation_name = c"VK_LAYER_KHRONOS_validation";
    let have_validation = std::env::var_os("PROBE_NO_VALIDATION").is_none()
        && entry
            .enumerate_instance_layer_properties()
            .check("enumerate layers")
            .iter()
            .any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == validation_name);
    let mut inst_exts = ash_window::enumerate_required_extensions(display).check("surface extensions").to_vec();
    inst_exts.push(ash::ext::debug_utils::NAME.as_ptr());
    let layers = if have_validation { vec![validation_name.as_ptr()] } else { vec![] };
    let app = vk::ApplicationInfo::default().application_name(c"window_probe").api_version(vk::API_VERSION_1_3);
    let instance = entry
        .create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app).enabled_layer_names(&layers).enabled_extension_names(&inst_exts),
            None,
        )
        .check("vkCreateInstance");
    let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
    let messenger = debug_utils
        .create_debug_utils_messenger(
            &vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_cb)),
            None,
        )
        .check("create messenger");
    let surface = ash_window::create_surface(&entry, &instance, display, whandle, None).check("create surface");
    let surface_ext = ash::khr::surface::Instance::new(&entry, &instance);

    // Device: RT-capable, and a graphics queue family that can present to this surface.
    let required = [
        ash::khr::acceleration_structure::NAME,
        ash::khr::ray_tracing_pipeline::NAME,
        ash::khr::deferred_host_operations::NAME,
        ash::khr::swapchain::NAME,
    ];
    let mut pick: Option<(vk::PhysicalDevice, u32, i32)> = None;
    for p in instance.enumerate_physical_devices().check("enumerate devices") {
        let exts = instance.enumerate_device_extension_properties(p).check("device exts");
        if !required.iter().all(|w| exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == *w)) {
            continue;
        }
        let qf = instance.get_physical_device_queue_family_properties(p);
        let Some(q) = (0..qf.len() as u32).find(|&i| {
            qf[i as usize].queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && surface_ext.get_physical_device_surface_support(p, i, surface).unwrap_or(false)
        }) else {
            continue;
        };
        let score = if instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU { 2 } else { 1 };
        if pick.map_or(true, |(_, _, s)| score > s) {
            pick = Some((p, q, score));
        }
    }
    let (phys, qfi, _) = pick.unwrap_or_else(|| fail("no RT device that can present to the window"));
    let pr = instance.get_physical_device_properties(phys);
    let device_name = CStr::from_ptr(pr.device_name.as_ptr()).to_string_lossy().into_owned();
    let mut rtp = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rtp);
    instance.get_physical_device_properties2(phys, &mut p2);
    let handle_size = rtp.shader_group_handle_size;

    let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
    let mut fas = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut frt = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
    let prio = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfi).queue_priorities(&prio)];
    let ext_ptrs: Vec<_> = required.iter().map(|e| e.as_ptr()).collect();
    let dev = instance
        .create_device(
            phys,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&qci)
                .enabled_extension_names(&ext_ptrs)
                .push_next(&mut f12)
                .push_next(&mut fas)
                .push_next(&mut frt),
            None,
        )
        .check("vkCreateDevice");
    let queue = dev.get_device_queue(qfi, 0);

    // Swapchain: FIFO (always supported), images cleared by transfer.
    let caps = surface_ext.get_physical_device_surface_capabilities(phys, surface).check("surface caps");
    let formats = surface_ext.get_physical_device_surface_formats(phys, surface).check("surface formats");
    let format = formats
        .iter()
        .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
        .copied()
        .unwrap_or(formats[0]);
    if !caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        fail("swapchain images cannot be transfer destinations");
    }
    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        let s = window.inner_size();
        vk::Extent2D { width: s.width, height: s.height }
    };
    let image_count = if caps.max_image_count == 0 { caps.min_image_count + 1 } else { (caps.min_image_count + 1).min(caps.max_image_count) };
    let present_mode = vk::PresentModeKHR::FIFO;
    let swapchain_ext = ash::khr::swapchain::Device::new(&instance, &dev);
    let swapchain = swapchain_ext
        .create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true),
            None,
        )
        .check("vkCreateSwapchainKHR");
    let images = swapchain_ext.get_swapchain_images(swapchain).check("swapchain images");

    let pool = dev
        .create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(qfi)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
        .check("vkCreateCommandPool");
    let cmd = dev
        .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1))
        .check("allocate cmd")[0];
    let fence = dev
        .create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
        .check("vkCreateFence");
    let acquired = dev.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).check("semaphore");
    // One "rendered" semaphore per swapchain image: presentation may still hold the previous one.
    let rendered = images
        .iter()
        .map(|_| dev.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).check("semaphore"))
        .collect();

    // Pipeline inputs (same interface as rt_probe, so the same SPIR-V is valid).
    let binds = [
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
    let dsl = dev
        .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds), None)
        .check("descriptor set layout");
    let pcr = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        offset: 0,
        size: 80,
    }];
    let dsls = [dsl];
    let layout = dev
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None)
        .check("pipeline layout");
    let files = ["rgen.spv", "miss.spv", "shadow.spv", "chit.spv"];
    let modules = files.map(|f| {
        let path = format!("{}/{f}", cfg.shader_dir);
        let mut file = std::fs::File::open(&path).unwrap_or_else(|_| fail(&format!("cannot open {path}")));
        let words = ash::util::read_spv(&mut file).unwrap_or_else(|_| fail(&format!("bad spv {path}")));
        dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None).check("shader module")
    });
    let seed = if cfg.unique_seed {
        // Nanoseconds since the epoch, never the sentinel value the shader tests for.
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos() | 1;
        n.min(0xFFFF_FFFE)
    } else {
        1
    };
    let job = CompileJob {
        rt: ash::khr::ray_tracing_pipeline::Device::new(&instance, &dev),
        modules,
        layout,
        seed,
        handle_size,
    };
    Gpu {
        _entry: entry,
        instance,
        debug_utils,
        messenger,
        surface_ext,
        surface,
        dev,
        swapchain_ext,
        swapchain,
        images,
        queue,
        pool,
        cmd,
        fence,
        acquired,
        rendered,
        job,
        dsl,
        device_name,
        present_mode,
    }
}

impl Gpu {
    unsafe fn present_frame(&self, t: f32) {
        let dev = &self.dev;
        dev.wait_for_fences(&[self.fence], true, u64::MAX).check("wait fence");
        dev.reset_fences(&[self.fence]).check("reset fence");
        let (index, _) = self
            .swapchain_ext
            .acquire_next_image(self.swapchain, u64::MAX, self.acquired, vk::Fence::null())
            .check("acquire");
        let image = self.images[index as usize];
        dev.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).check("reset cmd");
        dev.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
            .check("begin");
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let to_dst = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        dev.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_dst],
        );
        let color = vk::ClearColorValue { float32: [0.5 + 0.5 * (t * 3.0).sin(), 0.3, 0.5 + 0.5 * (t * 2.0).cos(), 1.0] };
        dev.cmd_clear_color_image(self.cmd, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &color, &[range]);
        let to_present = to_dst
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        dev.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_present],
        );
        dev.end_command_buffer(self.cmd).check("end");
        let waits = [self.acquired];
        let stages = [vk::PipelineStageFlags::TRANSFER];
        let signals = [self.rendered[index as usize]];
        let cmds = [self.cmd];
        dev.queue_submit(
            self.queue,
            &[vk::SubmitInfo::default()
                .wait_semaphores(&waits)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&cmds)
                .signal_semaphores(&signals)],
            self.fence,
        )
        .check("submit");
        let swapchains = [self.swapchain];
        let indices = [index];
        self.swapchain_ext
            .queue_present(self.queue, &vk::PresentInfoKHR::default().wait_semaphores(&signals).swapchains(&swapchains).image_indices(&indices))
            .check("present");
    }

    unsafe fn destroy(&self, pipeline: vk::Pipeline) {
        let dev = &self.dev;
        dev.device_wait_idle().check("wait idle");
        dev.destroy_pipeline(pipeline, None);
        dev.destroy_pipeline_layout(self.job.layout, None);
        for m in self.job.modules {
            dev.destroy_shader_module(m, None);
        }
        dev.destroy_descriptor_set_layout(self.dsl, None);
        for &s in &self.rendered {
            dev.destroy_semaphore(s, None);
        }
        dev.destroy_semaphore(self.acquired, None);
        dev.destroy_fence(self.fence, None);
        dev.destroy_command_pool(self.pool, None);
        self.swapchain_ext.destroy_swapchain(self.swapchain, None);
        dev.destroy_device(None);
        self.surface_ext.destroy_surface(self.surface, None);
        self.debug_utils.destroy_debug_utils_messenger(self.messenger, None);
        self.instance.destroy_instance(None);
    }
}

struct App {
    cfg: Config,
    window: Option<Window>,
    gpu: Option<Gpu>,
    t0: Instant,
    frames: Vec<(Instant, Instant)>, // (previous frame start, this frame start)
    last: Option<Instant>,
    started: bool,
    rx: Option<mpsc::Receiver<CompileResult>>,
    result: Option<CompileResult>,
}

impl App {
    fn report(&self) {
        let r = self.result.as_ref().unwrap();
        let stats = |sel: &dyn Fn(Instant, Instant) -> bool| {
            let mut v: Vec<f64> = self.frames.iter().filter(|(a, b)| sel(*a, *b)).map(|(a, b)| (*b - *a).as_secs_f64() * 1e3).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if v.is_empty() {
                return "{\"n\":0}".to_string();
            }
            let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
            format!("{{\"n\":{},\"median_ms\":{:.2},\"p99_ms\":{:.2},\"max_ms\":{:.2}}}", v.len(), q(0.5), q(0.99), v[v.len() - 1])
        };
        let (cs, ce) = (r.start, r.end);
        let gpu = self.gpu.as_ref().unwrap();
        println!(
            "{{\"probe\":\"window\",\"device\":\"{}\",\"present_mode\":\"{:?}\",\"compile_on\":\"{}\",\"seed\":\"{}\",\"seed_value\":{},\
\"compile_ms\":{:.2},\"handles_ok\":{},\"frames\":{},\"baseline\":{},\"during_compile\":{},\"after\":{},\
\"validation_errors\":{},\"validation_warnings\":{}}}",
            gpu.device_name,
            gpu.present_mode,
            if self.cfg.compile_on == CompileOn::Main { "main" } else { "background" },
            if self.cfg.unique_seed { "unique" } else { "fixed" },
            gpu.job.seed,
            (ce - cs).as_secs_f64() * 1e3,
            r.handles_ok,
            self.frames.len(),
            stats(&|_, b| b < cs),
            stats(&|a, b| b >= cs && a <= ce),
            stats(&|a, _| a > ce),
            VALIDATION_ERRORS.load(Ordering::Relaxed),
            VALIDATION_WARNINGS.load(Ordering::Relaxed),
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("phase0 window_probe")
                    .with_inner_size(PhysicalSize::new(1280, 720))
                    .with_resizable(false),
            )
            .unwrap_or_else(|e| fail(&format!("create window: {e}")));
        self.gpu = Some(unsafe { init_gpu(&window, &self.cfg) });
        self.window = Some(window);
        self.t0 = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => fail("window closed before the probe finished"),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                if let Some(last) = self.last {
                    self.frames.push((last, now));
                }
                self.last = Some(now);
                let gpu = self.gpu.as_ref().unwrap();
                if !self.started && now - self.t0 >= self.cfg.delay {
                    self.started = true;
                    let job = gpu.job.clone();
                    match self.cfg.compile_on {
                        CompileOn::Main => self.result = Some(compile(&job)),
                        CompileOn::Background => {
                            let (tx, rx) = mpsc::channel();
                            std::thread::spawn(move || {
                                let _ = tx.send(compile(&job));
                            });
                            self.rx = Some(rx);
                        }
                    }
                }
                if let Some(rx) = &self.rx {
                    if let Ok(r) = rx.try_recv() {
                        self.result = Some(r);
                        self.rx = None;
                    }
                }
                if let Some(r) = &self.result {
                    if now - r.end >= self.cfg.after {
                        self.report();
                        let pipeline = r.pipeline;
                        unsafe { self.gpu.as_ref().unwrap().destroy(pipeline) };
                        self.gpu = None;
                        event_loop.exit();
                        return;
                    }
                }
                unsafe { self.gpu.as_ref().unwrap().present_frame((now - self.t0).as_secs_f32()) };
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: window_probe <shader_dir> [--compile main|background] [--seed unique|fixed] [--delay-ms N] [--after-ms N]");
        std::process::exit(2);
    }
    let mut cfg = Config {
        shader_dir: args[1].clone(),
        compile_on: CompileOn::Background,
        unique_seed: true,
        delay: Duration::from_millis(1500),
        after: Duration::from_millis(1000),
    };
    let mut i = 2;
    while i + 1 < args.len() {
        match (args[i].as_str(), args[i + 1].as_str()) {
            ("--compile", "main") => cfg.compile_on = CompileOn::Main,
            ("--compile", "background") => cfg.compile_on = CompileOn::Background,
            ("--seed", "unique") => cfg.unique_seed = true,
            ("--seed", "fixed") => cfg.unique_seed = false,
            ("--delay-ms", v) => cfg.delay = Duration::from_millis(v.parse().unwrap_or_else(|_| fail("bad --delay-ms"))),
            ("--after-ms", v) => cfg.after = Duration::from_millis(v.parse().unwrap_or_else(|_| fail("bad --after-ms"))),
            (a, b) => fail(&format!("unknown argument {a} {b}")),
        }
        i += 2;
    }
    if i != args.len() {
        fail("arguments must come in pairs");
    }
    let event_loop = EventLoop::new().unwrap_or_else(|e| fail(&format!("event loop: {e}")));
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        cfg,
        window: None,
        gpu: None,
        t0: Instant::now(),
        frames: Vec::new(),
        last: None,
        started: false,
        rx: None,
        result: None,
    };
    event_loop.run_app(&mut app).unwrap_or_else(|e| fail(&format!("run: {e}")));
    let errors = VALIDATION_ERRORS.load(Ordering::Relaxed);
    std::process::exit(if errors > 0 { 3 } else { 0 });
}
