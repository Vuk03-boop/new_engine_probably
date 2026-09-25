//! Submission ordering and retirement on one timeline semaphore.
//!
//! - Every queue submission that uses resources signals the next timeline value. A resource (or a
//!   1C reader token) used by submissions up to value `v` is retired at `v`: it is freed (or
//!   released) only once the semaphore's counter has reached `v`.
//! - This replaces 1C's simulated reader tokens with real frames in flight: [`FrameReaders`] holds
//!   a `derived::ReaderToken` until the submission that read that snapshot has completed, so
//!   1C's last-reader retirement waits for the GPU.

use std::collections::VecDeque;

use ash::vk;

use crate::context::{Gpu, Result, VkCheck};

pub struct Timeline {
    pub semaphore: vk::Semaphore,
    last_signal: u64,
}

impl Timeline {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let mut t = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
        let semaphore = unsafe { gpu.device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut t), None) }.vk("vkCreateSemaphore")?;
        Ok(Self { semaphore, last_signal: 0 })
    }

    /// The value the next submission will signal. Values only grow.
    pub fn next_value(&mut self) -> u64 {
        self.last_signal += 1;
        self.last_signal
    }

    /// Highest value handed out so far.
    pub fn last_signal(&self) -> u64 {
        self.last_signal
    }

    /// Current counter value (completed work).
    pub fn completed(&self, gpu: &Gpu) -> Result<u64> {
        unsafe { gpu.device.get_semaphore_counter_value(self.semaphore) }.vk("vkGetSemaphoreCounterValue")
    }

    /// Blocks until the counter reaches `value`, or `timeout_ns` passes (`Ok(false)`).
    pub fn wait(&self, gpu: &Gpu, value: u64, timeout_ns: u64) -> Result<bool> {
        let sems = [self.semaphore];
        let vals = [value];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&vals);
        match unsafe { gpu.device.wait_semaphores(&info, timeout_ns) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(crate::context::GpuError::Vk { call: "vkWaitSemaphores", result: e }),
        }
    }

    /// Signals `value` from the host (used as a gate in tests).
    pub fn signal_from_host(&self, gpu: &Gpu, value: u64) -> Result<()> {
        unsafe { gpu.device.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(self.semaphore).value(value)) }.vk("vkSignalSemaphore")
    }

    pub fn destroy(self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_semaphore(self.semaphore, None) };
    }
}

/// Items waiting for the timeline to pass the value of their last use.
#[derive(Debug)]
pub struct Retirement<T> {
    pending: VecDeque<(u64, T)>,
}

impl<T> Default for Retirement<T> {
    fn default() -> Self {
        Self { pending: VecDeque::new() }
    }
}

impl<T> Retirement<T> {
    /// Retires `item` at `value`. Values must not decrease (they come from one timeline).
    pub fn push(&mut self, value: u64, item: T) {
        debug_assert!(self.pending.back().is_none_or(|(v, _)| *v <= value), "retirement values must not decrease");
        self.pending.push_back((value, item));
    }

    /// Removes and returns every item whose value is `<= completed`.
    pub fn collect(&mut self, completed: u64) -> Vec<T> {
        let mut out = Vec::new();
        while self.pending.front().is_some_and(|(v, _)| *v <= completed) {
            out.push(self.pending.pop_front().expect("front exists").1);
        }
        out
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

/// 1C reader tokens held by GPU submissions.
#[derive(Debug, Default)]
pub struct FrameReaders {
    held: Retirement<derived::ReaderToken>,
}

impl FrameReaders {
    /// The submission signalling `value` reads the snapshot `token` names.
    pub fn hold(&mut self, value: u64, token: derived::ReaderToken) {
        self.held.push(value, token);
    }

    /// Releases every token whose submission has completed. Returns how many were released.
    pub fn release_completed(&mut self, pipeline: &mut derived::Pipeline, completed: u64) -> usize {
        let done = self.held.collect(completed);
        let n = done.len();
        for t in done {
            pipeline.release(t).expect("token issued by this pipeline");
        }
        n
    }

    pub fn len(&self) -> usize {
        self.held.len()
    }

    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retirement_releases_in_order_only_after_completion() {
        let mut r = Retirement::default();
        r.push(3, "a");
        r.push(3, "b");
        r.push(5, "c");
        assert!(r.collect(2).is_empty());
        assert_eq!(r.collect(4), vec!["a", "b"]);
        assert_eq!(r.len(), 1);
        assert_eq!(r.collect(9), vec!["c"]);
        assert!(r.is_empty());
    }
}
