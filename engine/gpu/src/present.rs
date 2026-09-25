//! Phase 2C-2: swapchain and frames in flight on the 2B timeline.
//!
//! - Present mode: FIFO by default (always supported). MAILBOX or IMMEDIATE are used only when
//!   requested and supported; otherwise FIFO, and [`Swapchain::present_mode`] says which one runs.
//! - Format: `B8G8R8A8_SRGB` / sRGB-nonlinear when offered (the debug views write linear colour),
//!   otherwise the first format the surface lists.
//! - Swapchain images are driver-owned: they are **not** in the ADR-0003 ledger. Their size is
//!   reported by [`Swapchain::estimated_bytes`] (images × width × height × 4) for the record.
//! - [`Frames`] has one slot per frame in flight. A slot owns two command buffers and the binary
//!   semaphore its acquire signals, and remembers the timeline value of its last submission.
//!   [`Frames::begin`] waits for that value before reusing the slot, so at most `slots` frames are
//!   in flight and everything a slot's frame used (its command buffers, its timer queries, the 1C
//!   reader token it holds) is free once `begin` returns for that slot.
//! - A frame is two submissions. The **scene** command buffer (everything that does not touch the
//!   swapchain image, e.g. the G-buffer) is submitted by [`Frames::acquire`] *before* it acquires
//!   the image, with no semaphore wait. The **present** command buffer (drawing into the image) is
//!   submitted by [`Frames::submit_present`], waiting for the acquire at colour output. A semaphore
//!   wait at a stage holds that stage for every command in its batch, so one submission would make
//!   the G-buffer's colour writes wait for the swapchain image too.
//! - "Rendered" semaphores are per swapchain image (presentation may still hold the previous
//!   one when the slot comes round again).
//! - When the `Gpu` has present wait ([`Gpu::present_wait`]), every present carries an id (1, 2, …
//!   per swapchain) and [`Swapchain::wait_presented`] waits until an earlier present has reached
//!   the display. The viewer uses it as an optional pacing mode (M1 60 fps gate).

use std::cell::Cell;

use ash::vk;

use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::timeline::Timeline;

pub struct Swapchain {
    loader: ash::khr::swapchain::Device,
    pub handle: vk::SwapchainKHR,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    rendered: Vec<vk::Semaphore>,
    waiter: Option<ash::khr::present_wait::Device>,
    /// Id of the last present on this swapchain (0: none yet).
    last_present: Cell<u64>,
}

impl Swapchain {
    /// A swapchain for the `Gpu`'s surface. `window` is the window's size in pixels, used when the
    /// surface does not fix the extent. `Ok(None)` when the surface has zero area (minimized).
    /// `old` is retired into the new swapchain and destroyed; the caller must have waited for the
    /// GPU to finish with it.
    pub fn new(gpu: &Gpu, window: vk::Extent2D, want: vk::PresentModeKHR, old: Option<Swapchain>) -> Result<Option<Swapchain>> {
        let (surf, surface) = gpu.surface().expect("Gpu::with_surface");
        let caps = unsafe { surf.get_physical_device_surface_capabilities(gpu.physical, surface) }.vk("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")?;
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: window.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: window.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            if let Some(o) = old {
                o.destroy(gpu);
            }
            return Ok(None);
        }
        let formats = unsafe { surf.get_physical_device_surface_formats(gpu.physical, surface) }.vk("vkGetPhysicalDeviceSurfaceFormatsKHR")?;
        let format = formats
            .iter()
            .copied()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or(formats.first().copied())
            .ok_or(GpuError::NoDevice("surface lists no formats".into()))?;
        let modes = unsafe { surf.get_physical_device_surface_present_modes(gpu.physical, surface) }.vk("vkGetPhysicalDeviceSurfacePresentModesKHR")?;
        let present_mode = if modes.contains(&want) { want } else { vk::PresentModeKHR::FIFO };
        let count = if caps.max_image_count == 0 { caps.min_image_count + 1 } else { (caps.min_image_count + 1).min(caps.max_image_count) };
        let loader = ash::khr::swapchain::Device::new(&gpu.instance, &gpu.device);
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old.as_ref().map_or(vk::SwapchainKHR::null(), |o| o.handle));
        let handle = unsafe { loader.create_swapchain(&info, None) };
        if let Some(o) = old {
            o.destroy(gpu);
        }
        let handle = handle.vk("vkCreateSwapchainKHR")?;
        let images = unsafe { loader.get_swapchain_images(handle) }.vk("vkGetSwapchainImagesKHR")?;
        let mut views = Vec::new();
        let mut rendered = Vec::new();
        for &image in &images {
            let vi = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            views.push(unsafe { gpu.device.create_image_view(&vi, None) }.vk("vkCreateImageView")?);
            rendered.push(unsafe { gpu.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.vk("vkCreateSemaphore")?);
        }
        let waiter = gpu.present_wait().then(|| ash::khr::present_wait::Device::new(&gpu.instance, &gpu.device));
        Ok(Some(Swapchain { loader, handle, format, extent, present_mode, images, views, rendered, waiter, last_present: Cell::new(0) }))
    }

    /// Waits until every present on this swapchain except the last `pending` has reached the display,
    /// for at most `timeout_ns`. `None` without present wait; otherwise whether that happened in time
    /// (false also when the swapchain is out of date or the surface is lost).
    pub fn wait_presented(&self, pending: u64, timeout_ns: u64) -> Option<bool> {
        let w = self.waiter.as_ref()?;
        let target = self.last_present.get().saturating_sub(pending);
        if target == 0 {
            return Some(true);
        }
        Some(unsafe { w.wait_for_present(self.handle, target, timeout_ns) }.is_ok())
    }

    /// Driver-owned image bytes, estimated at 4 bytes per pixel (not in the ledger).
    pub fn estimated_bytes(&self) -> u64 {
        self.images.len() as u64 * self.extent.width as u64 * self.extent.height as u64 * 4
    }

    /// Destroys the swapchain; the GPU must be done with it.
    pub fn destroy(self, gpu: &Gpu) {
        unsafe {
            for v in self.views {
                gpu.device.destroy_image_view(v, None);
            }
            for s in self.rendered {
                gpu.device.destroy_semaphore(s, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}

struct Slot {
    pool: vk::CommandPool,
    /// [scene, present]
    cmds: [vk::CommandBuffer; 2],
    acquired: vk::Semaphore,
    /// Timeline value of this slot's last submission (0: none yet).
    value: u64,
}

/// A frame's scene command buffer, recording.
pub struct SceneCtx {
    pub slot: u32,
    pub cmd: vk::CommandBuffer,
}

/// A frame's present command buffer, recording, and the acquired image.
pub struct FrameCtx {
    pub slot: u32,
    pub cmd: vk::CommandBuffer,
    pub image: u32,
}

/// What [`Frames::acquire`] found. Either way the scene was submitted, at `scene_value`.
pub enum Acquired {
    /// Record into `frame`, then call [`Frames::submit_present`].
    Frame { frame: FrameCtx, scene_value: u64 },
    /// The swapchain no longer matches the surface: recreate it (after [`Frames::wait_all`]).
    OutOfDate { scene_value: u64 },
}

pub struct Frames {
    slots: Vec<Slot>,
    next: usize,
}

fn begin_cmd(gpu: &Gpu, cmd: vk::CommandBuffer) -> Result<()> {
    unsafe {
        gpu.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).vk("vkResetCommandBuffer")?;
        gpu.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).vk("vkBeginCommandBuffer")
    }
}

impl Frames {
    pub fn new(gpu: &Gpu, in_flight: usize) -> Result<Frames> {
        assert!(in_flight > 0);
        let mut slots = Vec::new();
        for _ in 0..in_flight {
            let info = vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let pool = unsafe { gpu.device.create_command_pool(&info, None) }.vk("vkCreateCommandPool")?;
            let ai = vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(2);
            let c = unsafe { gpu.device.allocate_command_buffers(&ai) }.vk("vkAllocateCommandBuffers")?;
            let acquired = unsafe { gpu.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.vk("vkCreateSemaphore")?;
            slots.push(Slot { pool, cmds: [c[0], c[1]], acquired, value: 0 });
        }
        Ok(Frames { slots, next: 0 })
    }

    pub fn in_flight(&self) -> usize {
        self.slots.len()
    }

    /// Waits until the next slot's previous frame has completed and begins its scene command
    /// buffer.
    pub fn begin(&mut self, gpu: &Gpu, timeline: &Timeline) -> Result<SceneCtx> {
        let s = &self.slots[self.next];
        if s.value > 0 {
            timeline.wait(gpu, s.value, u64::MAX)?;
        }
        begin_cmd(gpu, s.cmds[0])?;
        Ok(SceneCtx { slot: self.next as u32, cmd: s.cmds[0] })
    }

    /// Submits the scene (no semaphore wait; signals the next timeline value), then acquires a
    /// swapchain image and begins the present command buffer.
    pub fn acquire(&mut self, gpu: &Gpu, timeline: &mut Timeline, swapchain: &Swapchain, scene: SceneCtx) -> Result<Acquired> {
        let s = &mut self.slots[scene.slot as usize];
        unsafe { gpu.device.end_command_buffer(scene.cmd) }.vk("vkEndCommandBuffer")?;
        let scene_value = timeline.next_value();
        let signals = [vk::SemaphoreSubmitInfo::default().semaphore(timeline.semaphore).value(scene_value).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(scene.cmd)];
        let submit = [vk::SubmitInfo2::default().command_buffer_infos(&cmds).signal_semaphore_infos(&signals)];
        unsafe { gpu.device.queue_submit2(gpu.queue, &submit, vk::Fence::null()) }.vk("vkQueueSubmit2")?;
        s.value = scene_value;
        let image = match unsafe { swapchain.loader.acquire_next_image(swapchain.handle, u64::MAX, s.acquired, vk::Fence::null()) } {
            Ok((i, _suboptimal)) => i,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(Acquired::OutOfDate { scene_value }),
            Err(e) => return Err(GpuError::Vk { call: "vkAcquireNextImageKHR", result: e }),
        };
        begin_cmd(gpu, s.cmds[1])?;
        Ok(Acquired::Frame { frame: FrameCtx { slot: scene.slot, cmd: s.cmds[1], image }, scene_value })
    }

    /// Ends and submits the present command buffer (waiting for its acquire at colour output,
    /// signalling the next timeline value and the image's "rendered" semaphore), then presents.
    /// Returns the timeline value and whether the swapchain should be recreated (suboptimal or out
    /// of date).
    pub fn submit_present(&mut self, gpu: &Gpu, timeline: &mut Timeline, swapchain: &Swapchain, frame: FrameCtx) -> Result<(u64, bool)> {
        let s = &mut self.slots[frame.slot as usize];
        unsafe { gpu.device.end_command_buffer(frame.cmd) }.vk("vkEndCommandBuffer")?;
        let value = timeline.next_value();
        let rendered = swapchain.rendered[frame.image as usize];
        let waits = [vk::SemaphoreSubmitInfo::default().semaphore(s.acquired).stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let signals = [
            vk::SemaphoreSubmitInfo::default().semaphore(timeline.semaphore).value(value).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::SemaphoreSubmitInfo::default().semaphore(rendered).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.cmd)];
        let submit = [vk::SubmitInfo2::default().wait_semaphore_infos(&waits).command_buffer_infos(&cmds).signal_semaphore_infos(&signals)];
        unsafe { gpu.device.queue_submit2(gpu.queue, &submit, vk::Fence::null()) }.vk("vkQueueSubmit2")?;
        s.value = value;
        self.next = (self.next + 1) % self.slots.len();
        let sw = [swapchain.handle];
        let idx = [frame.image];
        let ws = [rendered];
        let id = [swapchain.last_present.get() + 1];
        let mut present_id = vk::PresentIdKHR::default().present_ids(&id);
        let mut pi = vk::PresentInfoKHR::default().wait_semaphores(&ws).swapchains(&sw).image_indices(&idx);
        if swapchain.waiter.is_some() {
            pi = pi.push_next(&mut present_id);
            swapchain.last_present.set(id[0]);
        }
        let recreate = match unsafe { swapchain.loader.queue_present(gpu.queue, &pi) } {
            Ok(suboptimal) => suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(e) => return Err(GpuError::Vk { call: "vkQueuePresentKHR", result: e }),
        };
        Ok((value, recreate))
    }

    /// Waits for every slot's last submission.
    pub fn wait_all(&self, gpu: &Gpu, timeline: &Timeline) -> Result<()> {
        for s in &self.slots {
            if s.value > 0 {
                timeline.wait(gpu, s.value, u64::MAX)?;
            }
        }
        Ok(())
    }

    /// Destroys the slots. Presentation may still hold semaphores, so the device is idled first.
    pub fn destroy(self, gpu: &Gpu) {
        let _ = gpu.wait_idle();
        for s in self.slots {
            unsafe {
                gpu.device.destroy_semaphore(s.acquired, None);
                gpu.device.destroy_command_pool(s.pool, None);
            }
        }
    }
}

/// Transitions a swapchain image for rendering (contents discarded). Its source stage is colour
/// output, which the acquire semaphore wait covers.
pub fn image_to_attachment(gpu: &Gpu, cmd: vk::CommandBuffer, image: vk::Image) {
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })];
    unsafe { gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
}

/// Transitions a rendered swapchain image for presentation.
pub fn image_to_present(gpu: &Gpu, cmd: vk::CommandBuffer, image: vk::Image) {
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::NONE)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })];
    unsafe { gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
}
