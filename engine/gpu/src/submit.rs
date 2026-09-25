//! Command buffers and queue submission on the shared timeline.

use ash::vk;

use crate::context::{Gpu, Result, VkCheck};
use crate::timeline::{Retirement, Timeline};

/// Recycles command buffers once the submission that used them has completed.
pub struct Submitter {
    pool: vk::CommandPool,
    free: Vec<vk::CommandBuffer>,
    in_flight: Retirement<vk::CommandBuffer>,
}

impl Submitter {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { gpu.device.create_command_pool(&info, None) }.vk("vkCreateCommandPool")?;
        Ok(Self { pool, free: Vec::new(), in_flight: Retirement::default() })
    }

    /// A command buffer in the recording state.
    pub fn begin(&mut self, gpu: &Gpu, timeline: &Timeline) -> Result<vk::CommandBuffer> {
        let done = timeline.completed(gpu)?;
        self.free.extend(self.in_flight.collect(done));
        let cmd = match self.free.pop() {
            Some(c) => {
                unsafe { gpu.device.reset_command_buffer(c, vk::CommandBufferResetFlags::empty()) }.vk("vkResetCommandBuffer")?;
                c
            }
            None => {
                let ai = vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
                unsafe { gpu.device.allocate_command_buffers(&ai) }.vk("vkAllocateCommandBuffers")?[0]
            }
        };
        let bi = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { gpu.device.begin_command_buffer(cmd, &bi) }.vk("vkBeginCommandBuffer")?;
        Ok(cmd)
    }

    /// Ends and submits `cmd`. It waits for each `(semaphore, value)` in `waits` (timeline
    /// semaphores, all stages) and signals the timeline's next value, which is returned.
    pub fn submit(&mut self, gpu: &Gpu, timeline: &mut Timeline, cmd: vk::CommandBuffer, waits: &[(vk::Semaphore, u64)]) -> Result<u64> {
        unsafe { gpu.device.end_command_buffer(cmd) }.vk("vkEndCommandBuffer")?;
        let value = timeline.next_value();
        let wait_infos: Vec<_> = waits.iter().map(|&(s, v)| vk::SemaphoreSubmitInfo::default().semaphore(s).value(v).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)).collect();
        let signal = [vk::SemaphoreSubmitInfo::default().semaphore(timeline.semaphore).value(value).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submit = [vk::SubmitInfo2::default().wait_semaphore_infos(&wait_infos).command_buffer_infos(&cmds).signal_semaphore_infos(&signal)];
        unsafe { gpu.device.queue_submit2(gpu.queue, &submit, vk::Fence::null()) }.vk("vkQueueSubmit2")?;
        self.in_flight.push(value, cmd);
        Ok(value)
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_command_pool(self.pool, None) };
    }
}

/// Makes transfer writes visible to every later command on the queue.
pub fn transfer_to_all(gpu: &Gpu, cmd: vk::CommandBuffer) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    unsafe { gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
}

/// Makes all prior writes visible to host reads.
pub fn all_to_host(gpu: &Gpu, cmd: vk::CommandBuffer) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ)];
    unsafe { gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b)) };
}
