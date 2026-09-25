//! Phase 2C-2: GPU timestamps per pass, and percentile summaries of frame and pass times.
//!
//! - One query pool with `slots × passes × 2` timestamps: each pass writes its own begin and end
//!   stamp, so a pass's time never includes gaps between submissions (the viewer submits the scene
//!   and the view pass separately, with a swapchain acquire in between). Pass times are not nested
//!   and are never summed into a frame time; frame time is measured on the CPU.
//! - A slot's results are read only after the timeline shows its submission complete (the caller
//!   waits for that before reusing the slot), so reads never block and never see stale values:
//!   the slot is reset in the same command buffer before it is written again.
//! - Ticks are converted with `timestampPeriod`; the queue's `timestampValidBits` is checked at
//!   device selection (non-zero), and results are masked to those bits.

use ash::vk;

use crate::context::{Gpu, Result, VkCheck};

pub struct GpuTimer {
    pool: vk::QueryPool,
    slots: u32,
    passes: u32,
    /// Slot has been written by a submitted frame.
    written: Vec<bool>,
    period_ns: f64,
    valid_mask: u64,
}

impl GpuTimer {
    pub fn new(gpu: &Gpu, slots: u32, passes: u32) -> Result<GpuTimer> {
        let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(slots * passes * 2);
        let pool = unsafe { gpu.device.create_query_pool(&info, None) }.vk("vkCreateQueryPool")?;
        let bits = unsafe { gpu.instance.get_physical_device_queue_family_properties(gpu.physical) }[gpu.queue_family as usize].timestamp_valid_bits;
        let valid_mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        Ok(GpuTimer { pool, slots, passes, written: vec![false; slots as usize], period_ns: gpu.info.timestamp_period as f64, valid_mask })
    }

    fn base(&self, slot: u32) -> u32 {
        assert!(slot < self.slots);
        slot * self.passes * 2
    }

    /// Resets the slot's queries. Record once per frame, before the slot's first pass, in the
    /// frame's first command buffer.
    pub fn reset(&mut self, gpu: &Gpu, cmd: vk::CommandBuffer, slot: u32) {
        unsafe { gpu.device.cmd_reset_query_pool(cmd, self.pool, self.base(slot), self.passes * 2) };
        self.written[slot as usize] = true;
    }

    /// Writes the stamp that begins pass `pass` (0-based), before its work.
    pub fn begin_pass(&self, gpu: &Gpu, cmd: vk::CommandBuffer, slot: u32, pass: u32) {
        assert!(pass < self.passes);
        unsafe { gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, self.pool, self.base(slot) + 2 * pass) };
    }

    /// Writes the stamp that ends pass `pass`, after all its work.
    pub fn end_pass(&self, gpu: &Gpu, cmd: vk::CommandBuffer, slot: u32, pass: u32) {
        assert!(pass < self.passes);
        unsafe { gpu.device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, self.pool, self.base(slot) + 2 * pass + 1) };
    }

    /// Pass durations in milliseconds of the slot's last completed frame, or `None` if the slot has
    /// not been written yet. Call only after that frame's submission has completed.
    pub fn read(&self, gpu: &Gpu, slot: u32) -> Result<Option<Vec<f64>>> {
        if !self.written[slot as usize] {
            return Ok(None);
        }
        let mut stamps = vec![0u64; (self.passes * 2) as usize];
        unsafe { gpu.device.get_query_pool_results(self.pool, self.base(slot), &mut stamps, vk::QueryResultFlags::TYPE_64) }.vk("vkGetQueryPoolResults")?;
        let ms = stamps.chunks(2).map(|w| ((w[1] & self.valid_mask).wrapping_sub(w[0] & self.valid_mask) & self.valid_mask) as f64 * self.period_ns * 1e-6).collect();
        Ok(Some(ms))
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_query_pool(self.pool, None) };
    }
}

/// Nearest-rank percentile of `samples` (`p` in 0..=100). `None` for no samples.
pub fn percentile(samples: &[f64], p: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut s = samples.to_vec();
    s.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * s.len() as f64).ceil().max(1.0) as usize;
    Some(s[rank.min(s.len()) - 1])
}

#[cfg(test)]
mod tests {
    use super::percentile;

    #[test]
    fn nearest_rank_percentiles() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 50.0), Some(50.0));
        assert_eq!(percentile(&v, 99.0), Some(99.0));
        assert_eq!(percentile(&v, 100.0), Some(100.0));
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 0.0), Some(1.0));
        assert_eq!(percentile(&[], 50.0), None);
    }
}
