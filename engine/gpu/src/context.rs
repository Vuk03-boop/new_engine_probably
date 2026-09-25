//! Instance, validation, device selection and the device memory budget.
//!
//! - Validation (`VK_LAYER_KHRONOS_validation`) is on whenever the layer exists, unless
//!   `NE_NO_VALIDATION` is set. Messages are counted per context; tests assert zero errors.
//! - The device is chosen by capability, never by index: Vulkan 1.3, acceleration structures and
//!   ray query (P-001: ray tracing is required), a graphics+compute queue, and the features the
//!   raster path uses (`geometryShader` for fragment `PrimitiveID`, `dynamicRendering`). Discrete
//!   GPUs win ties.
//! - The device budget is `memory::device_budget` of the driver's device-local heap budget
//!   (`VK_EXT_memory_budget`, when present), read once at creation.
//! - With a window ([`Gpu::with_surface`], 2C-2): the caller's surface extensions are enabled, the
//!   surface is created from the new instance, and a device qualifies only if its queue family can
//!   present to that surface and it has `VK_KHR_swapchain`. The surface is owned by the `Gpu`.
//! - Diagnostic only: `NE_GPU_ALLOW_NO_RT=1` also accepts a device without ray tracing and leaves
//!   the RT features off ([`Gpu::ray_tracing`] is then false). It exists to exercise the non-RT
//!   parts (memory, uploads, retirement, compute) when the RT GPU is unavailable; results from it
//!   are supplemental and never count as the P-001 device gate.

use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use ash::vk;

#[derive(Debug)]
pub enum GpuError {
    Vk { call: &'static str, result: vk::Result },
    Load(String),
    NoDevice(String),
    OverBudget(memory::LedgerError),
    Layout(Vec<String>),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for GpuError {}

pub type Result<T> = std::result::Result<T, GpuError>;

pub(crate) trait VkCheck<T> {
    fn vk(self, call: &'static str) -> Result<T>;
}

impl<T> VkCheck<T> for std::result::Result<T, vk::Result> {
    fn vk(self, call: &'static str) -> Result<T> {
        self.map_err(|result| GpuError::Vk { call, result })
    }
}

/// Validation message counts. The first few error texts are kept for test failure messages.
#[derive(Default)]
pub struct Validation {
    pub errors: AtomicU64,
    pub warnings: AtomicU64,
    pub first_errors: Mutex<Vec<String>>,
}

unsafe extern "system" fn debug_cb(
    sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    let v = &*(user as *const Validation);
    let msg = if data.is_null() || (*data).p_message.is_null() { String::new() } else { CStr::from_ptr((*data).p_message).to_string_lossy().into_owned() };
    if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        v.errors.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut f) = v.first_errors.lock() {
            if f.len() < 8 {
                f.push(msg.clone());
            }
        }
        eprintln!("[validation error] {msg}");
    } else {
        v.warnings.fetch_add(1, Ordering::Relaxed);
        eprintln!("[validation warning] {msg}");
    }
    vk::FALSE
}

/// Device facts the rest of the crate needs.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub api_version: u32,
    pub driver_version: u32,
    pub min_storage_buffer_offset_alignment: u64,
    pub non_coherent_atom_size: u64,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    /// Sum of device-local heap budgets from `VK_EXT_memory_budget`, if the extension exists.
    pub driver_heap_budget: Option<u64>,
    pub memory: vk::PhysicalDeviceMemoryProperties,
}

pub struct Gpu {
    // Declared in drop order: the device before the instance.
    pub device: ash::Device,
    pub instance: ash::Instance,
    pub entry: ash::Entry,
    pub physical: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub info: DeviceInfo,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    surface: Option<(ash::khr::surface::Instance, vk::SurfaceKHR)>,
    validation: Box<Validation>,
    has_memory_budget: bool,
    ray_tracing: bool,
    present_wait: bool,
}

const REQUIRED_DEVICE_EXTENSIONS: [&CStr; 3] = [ash::khr::acceleration_structure::NAME, ash::khr::ray_query::NAME, ash::khr::deferred_host_operations::NAME];

impl Gpu {
    pub fn new() -> Result<Gpu> {
        unsafe { Self::create(&[], None) }
    }

    /// A device that can present: `extensions` are the instance extensions the window system needs
    /// (for example from `ash_window::enumerate_required_extensions`), and `make_surface` creates the
    /// surface from the new instance. The surface is destroyed with the `Gpu`; swapchains made on
    /// it must be destroyed first.
    pub fn with_surface(extensions: &[*const c_char], make_surface: impl FnOnce(&ash::Entry, &ash::Instance) -> std::result::Result<vk::SurfaceKHR, vk::Result>) -> Result<Gpu> {
        unsafe { Self::create(extensions, Some(Box::new(make_surface))) }
    }

    /// The window surface, when made by [`Gpu::with_surface`].
    pub fn surface(&self) -> Option<(&ash::khr::surface::Instance, vk::SurfaceKHR)> {
        self.surface.as_ref().map(|(l, s)| (l, *s))
    }

    #[allow(clippy::type_complexity)]
    unsafe fn create(extensions: &[*const c_char], make_surface: Option<Box<dyn FnOnce(&ash::Entry, &ash::Instance) -> std::result::Result<vk::SurfaceKHR, vk::Result> + '_>>) -> Result<Gpu> {
        let entry = ash::Entry::load().map_err(|e| GpuError::Load(e.to_string()))?;
        let layer = c"VK_LAYER_KHRONOS_validation";
        let want_validation = std::env::var_os("NE_NO_VALIDATION").is_none()
            && entry.enumerate_instance_layer_properties().vk("vkEnumerateInstanceLayerProperties")?.iter().any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == layer);
        let validation = Box::new(Validation::default());
        let dbg_info = || {
            vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
                .pfn_user_callback(Some(debug_cb))
                .user_data(&*validation as *const Validation as *mut c_void)
        };
        let app = vk::ApplicationInfo::default().application_name(c"new-engine").api_version(vk::API_VERSION_1_3);
        let layers = [layer.as_ptr()];
        let mut exts = extensions.to_vec();
        if want_validation {
            exts.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let mut chained = dbg_info();
        let mut ici = vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&exts);
        if want_validation {
            ici = ici.enabled_layer_names(&layers).push_next(&mut chained);
        }
        let instance = entry.create_instance(&ici, None).vk("vkCreateInstance")?;
        let debug = if want_validation {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let m = du.create_debug_utils_messenger(&dbg_info(), None).vk("vkCreateDebugUtilsMessengerEXT")?;
            Some((du, m))
        } else {
            None
        };
        let surface = match make_surface {
            None => None,
            Some(make) => match make(&entry, &instance) {
                Ok(s) => Some((ash::khr::surface::Instance::new(&entry, &instance), s)),
                Err(e) => {
                    if let Some((du, m)) = &debug {
                        du.destroy_debug_utils_messenger(*m, None);
                    }
                    instance.destroy_instance(None);
                    return Err(GpuError::Vk { call: "create surface", result: e });
                }
            },
        };

        // Capability-based selection.
        let allow_no_rt = std::env::var_os("NE_GPU_ALLOW_NO_RT").is_some();
        let mut best: Option<(i32, vk::PhysicalDevice, u32, bool, bool, bool)> = None;
        let mut rejected = Vec::new();
        for p in instance.enumerate_physical_devices().vk("vkEnumeratePhysicalDevices")? {
            let props = instance.get_physical_device_properties(p);
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
            let exts = instance.enumerate_device_extension_properties(p).vk("vkEnumerateDeviceExtensionProperties")?;
            let has = |want: &CStr| exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == want);
            let rt = REQUIRED_DEVICE_EXTENSIONS.iter().all(|e| has(e));
            if props.api_version < vk::API_VERSION_1_3 || !(rt || allow_no_rt) {
                rejected.push(format!("{name}: needs Vulkan 1.3, acceleration structures and ray query"));
                continue;
            }
            let mut q13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut qf = vk::PhysicalDeviceFeatures2::default().push_next(&mut q13);
            instance.get_physical_device_features2(p, &mut qf);
            if qf.features.geometry_shader == vk::FALSE || q13.dynamic_rendering == vk::FALSE {
                rejected.push(format!("{name}: needs geometryShader and dynamicRendering"));
                continue;
            }
            if surface.is_some() && !has(ash::khr::swapchain::NAME) {
                rejected.push(format!("{name}: no VK_KHR_swapchain"));
                continue;
            }
            let presents = |i: usize| surface.as_ref().is_none_or(|(l, s)| l.get_physical_device_surface_support(p, i as u32, *s).unwrap_or(false));
            let Some(family) = instance
                .get_physical_device_queue_family_properties(p)
                .iter()
                .enumerate()
                .position(|(i, q)| q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) && q.timestamp_valid_bits > 0 && presents(i))
            else {
                rejected.push(format!("{name}: no graphics+compute queue with timestamps{}", if surface.is_some() { " that presents to the window" } else { "" }));
                continue;
            };
            // Optional, presenting devices only: VK_KHR_present_id + VK_KHR_present_wait, so the
            // caller can wait until a present has reached the display.
            let present_wait = surface.is_some() && has(ash::khr::present_id::NAME) && has(ash::khr::present_wait::NAME) && {
                let mut qid = vk::PhysicalDevicePresentIdFeaturesKHR::default();
                let mut qpw = vk::PhysicalDevicePresentWaitFeaturesKHR::default();
                let mut qf = vk::PhysicalDeviceFeatures2::default().push_next(&mut qid).push_next(&mut qpw);
                instance.get_physical_device_features2(p, &mut qf);
                qid.present_id == vk::TRUE && qpw.present_wait == vk::TRUE
            };
            // RT devices always beat non-RT ones; then discrete beats integrated.
            let score = if rt { 10 } else { 0 } + if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU { 2 } else { 1 };
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, p, family as u32, has(ash::ext::memory_budget::NAME), rt, present_wait));
            }
        }
        let Some((_, physical, queue_family, has_memory_budget, ray_tracing, present_wait)) = best else {
            if let Some((l, s)) = &surface {
                l.destroy_surface(*s, None);
            }
            if let Some((du, m)) = &debug {
                du.destroy_debug_utils_messenger(*m, None);
            }
            instance.destroy_instance(None);
            return Err(GpuError::NoDevice(rejected.join("; ")));
        };

        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true).timeline_semaphore(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true).dynamic_rendering(true);
        let base = vk::PhysicalDeviceFeatures::default().geometry_shader(true);
        let mut fas = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
        let mut frq = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
        let mut ext_ptrs: Vec<_> = if ray_tracing { REQUIRED_DEVICE_EXTENSIONS.iter().map(|e| e.as_ptr()).collect() } else { Vec::new() };
        if has_memory_budget {
            ext_ptrs.push(ash::ext::memory_budget::NAME.as_ptr());
        }
        if surface.is_some() {
            ext_ptrs.push(ash::khr::swapchain::NAME.as_ptr());
        }
        if present_wait {
            ext_ptrs.push(ash::khr::present_id::NAME.as_ptr());
            ext_ptrs.push(ash::khr::present_wait::NAME.as_ptr());
        }
        let mut fid = vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);
        let mut fpw = vk::PhysicalDevicePresentWaitFeaturesKHR::default().present_wait(true);
        let mut dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci).enabled_extension_names(&ext_ptrs).enabled_features(&base).push_next(&mut f12).push_next(&mut f13);
        if ray_tracing {
            dci = dci.push_next(&mut fas).push_next(&mut frq);
        }
        if present_wait {
            dci = dci.push_next(&mut fid).push_next(&mut fpw);
        }
        let device = instance.create_device(physical, &dci, None).vk("vkCreateDevice")?;
        let queue = device.get_device_queue(queue_family, 0);

        let props = instance.get_physical_device_properties(physical);
        let mut gpu = Gpu {
            device,
            instance,
            entry,
            physical,
            queue,
            queue_family,
            info: DeviceInfo {
                name: CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned(),
                api_version: props.api_version,
                driver_version: props.driver_version,
                min_storage_buffer_offset_alignment: props.limits.min_storage_buffer_offset_alignment,
                non_coherent_atom_size: props.limits.non_coherent_atom_size,
                timestamp_period: props.limits.timestamp_period,
                driver_heap_budget: None,
                memory: vk::PhysicalDeviceMemoryProperties::default(),
            },
            debug,
            surface,
            validation,
            has_memory_budget,
            ray_tracing,
            present_wait,
        };
        gpu.info.memory = gpu.instance.get_physical_device_memory_properties(physical);
        gpu.info.driver_heap_budget = gpu.heap_usage().map(|(_, b)| b);
        Ok(gpu)
    }

    /// Device-local heap (usage, budget) from `VK_EXT_memory_budget`, summed over device-local heaps.
    pub fn heap_usage(&self) -> Option<(u64, u64)> {
        if !self.has_memory_budget {
            return None;
        }
        let mut mb = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut mp = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut mb);
        unsafe { self.instance.get_physical_device_memory_properties2(self.physical, &mut mp) };
        let heaps = mp.memory_properties;
        let (mut usage, mut budget) = (0, 0);
        for i in 0..heaps.memory_heap_count as usize {
            if heaps.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                usage += mb.heap_usage[i];
                budget += mb.heap_budget[i];
            }
        }
        Some((usage, budget))
    }

    /// ADR-0003 device budget for this device.
    pub fn device_budget(&self) -> memory::Budget {
        memory::device_budget(self.info.driver_heap_budget)
    }

    /// True when acceleration structures and ray query are enabled (always, unless the
    /// `NE_GPU_ALLOW_NO_RT` diagnostic selected a non-RT device).
    pub fn ray_tracing(&self) -> bool {
        self.ray_tracing
    }

    /// True when `VK_KHR_present_id` and `VK_KHR_present_wait` are enabled (presenting devices
    /// that support them): see [`crate::present::Swapchain::wait_presented`].
    pub fn present_wait(&self) -> bool {
        self.present_wait
    }

    pub fn validation_enabled(&self) -> bool {
        self.debug.is_some()
    }

    /// (errors, warnings) reported by validation so far.
    pub fn validation_counts(&self) -> (u64, u64) {
        (self.validation.errors.load(Ordering::Relaxed), self.validation.warnings.load(Ordering::Relaxed))
    }

    pub fn first_validation_errors(&self) -> Vec<String> {
        self.validation.first_errors.lock().map(|v| v.clone()).unwrap_or_default()
    }

    pub fn wait_idle(&self) -> Result<()> {
        unsafe { self.device.device_wait_idle() }.vk("vkDeviceWaitIdle")
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            if let Some((l, s)) = self.surface.take() {
                l.destroy_surface(s, None);
            }
            if let Some((du, m)) = self.debug.take() {
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
