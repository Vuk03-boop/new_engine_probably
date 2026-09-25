//! Host → device uploads through a staging ring, and device → host readback.
//!
//! The ring is one host-visible buffer (`Category::Staging`, one ledger grant). Data is copied in,
//! a `vkCmdCopyBuffer` is recorded, and the space is reused only after the submission that copied
//! out of it has completed on the timeline. Uploads larger than a quarter of the ring are split, so
//! any size fits. When the ring is full, the pending batch is submitted and the oldest in-flight
//! batch is waited for: back-pressure, never overwriting data the GPU has not read yet.

use std::collections::VecDeque;

use ash::vk;
use memory::Category;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, Result};
use crate::submit::{all_to_host, transfer_to_all, Submitter};
use crate::timeline::Timeline;

pub const DEFAULT_RING_BYTES: u64 = 16 << 20;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadStats {
    pub bytes: u64,
    pub copies: u64,
    pub batches: u64,
    /// Times the ring was full and the host waited for the GPU.
    pub stalls: u64,
}

pub struct Uploader {
    ring: Buffer,
    cap: u64,
    /// Total bytes ever reserved (monotonic); ring position is `head % cap`.
    head: u64,
    /// Total bytes known to be free again.
    tail: u64,
    /// (timeline value, head at submission) for batches in flight.
    spans: VecDeque<(u64, u64)>,
    cmd: Option<vk::CommandBuffer>,
    submitter: Submitter,
    stats: UploadStats,
}

impl Uploader {
    pub fn new(gpu: &Gpu, alloc: &mut Allocator, ring_bytes: u64) -> Result<Self> {
        let ring = alloc.create_buffer(gpu, ring_bytes, vk::BufferUsageFlags::TRANSFER_SRC, Category::Staging, Kind::Host)?;
        Ok(Self { ring, cap: ring_bytes, head: 0, tail: 0, spans: VecDeque::new(), cmd: None, submitter: Submitter::new(gpu)?, stats: UploadStats::default() })
    }

    pub fn stats(&self) -> UploadStats {
        self.stats
    }

    fn retire(&mut self, completed: u64) {
        while let Some(&(v, end)) = self.spans.front() {
            if v > completed {
                break;
            }
            self.tail = end;
            self.spans.pop_front();
        }
        if self.spans.is_empty() && self.cmd.is_none() {
            self.tail = self.head;
        }
    }

    /// Reserves `n` contiguous ring bytes, waiting for the GPU if needed. Returns the ring offset.
    fn reserve(&mut self, gpu: &Gpu, timeline: &mut Timeline, n: u64) -> Result<u64> {
        debug_assert!(n <= self.cap);
        loop {
            self.retire(timeline.completed(gpu)?);
            let pos = self.head % self.cap;
            let skip = if pos + n > self.cap { self.cap - pos } else { 0 };
            if self.cap - (self.head - self.tail) >= skip + n {
                self.head += skip;
                let at = self.head % self.cap;
                self.head += n;
                return Ok(at);
            }
            // Full: submit what is pending, then wait for the oldest batch.
            self.flush(gpu, timeline)?;
            let (v, _) = *self.spans.front().expect("a full ring has a batch in flight");
            self.stats.stalls += 1;
            timeline.wait(gpu, v, u64::MAX)?;
        }
    }

    /// Queues a copy of `bytes` into `dst` at `dst_offset`. It is visible to commands submitted
    /// after the next [`Uploader::flush`].
    pub fn upload(&mut self, gpu: &Gpu, timeline: &mut Timeline, dst: &Buffer, dst_offset: u64, bytes: &[u8]) -> Result<()> {
        assert!(dst_offset + bytes.len() as u64 <= dst.size, "upload past the end of the buffer");
        let chunk = (self.cap / 4).max(1) as usize;
        for (i, part) in bytes.chunks(chunk).enumerate() {
            let at = self.reserve(gpu, timeline, part.len() as u64)?;
            self.ring.mapped().expect("ring is mapped")[at as usize..at as usize + part.len()].copy_from_slice(part);
            if self.cmd.is_none() {
                self.cmd = Some(self.submitter.begin(gpu, timeline)?);
            }
            let region = [vk::BufferCopy { src_offset: at, dst_offset: dst_offset + (i * chunk) as u64, size: part.len() as u64 }];
            unsafe { gpu.device.cmd_copy_buffer(self.cmd.unwrap(), self.ring.buffer, dst.buffer, &region) };
            self.stats.copies += 1;
            self.stats.bytes += part.len() as u64;
        }
        Ok(())
    }

    /// Submits pending copies. Returns the timeline value after which they are complete (the last
    /// signalled value if nothing was pending).
    pub fn flush(&mut self, gpu: &Gpu, timeline: &mut Timeline) -> Result<u64> {
        let Some(cmd) = self.cmd.take() else { return Ok(timeline.last_signal()) };
        transfer_to_all(gpu, cmd);
        let v = self.submitter.submit(gpu, timeline, cmd, &[])?;
        self.spans.push_back((v, self.head));
        self.stats.batches += 1;
        Ok(v)
    }

    pub fn destroy(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free(gpu, self.ring);
        self.submitter.destroy(gpu);
    }
}

/// Copies `len` bytes of each `(src, offset, len)` back to the host in one submission and waits.
/// For tests and verification; it allocates one temporary host buffer.
pub fn download(gpu: &Gpu, alloc: &mut Allocator, timeline: &mut Timeline, ranges: &[(&Buffer, u64, u64)]) -> Result<Vec<Vec<u8>>> {
    let total: u64 = ranges.iter().map(|r| r.2).sum();
    let tmp = alloc.create_buffer(gpu, total.max(1), vk::BufferUsageFlags::TRANSFER_DST, Category::Staging, Kind::Host)?;
    let mut sub = Submitter::new(gpu)?;
    let result = (|| {
        let cmd = sub.begin(gpu, timeline)?;
        let mut at = 0;
        for &(src, offset, len) in ranges {
            if len > 0 {
                let region = [vk::BufferCopy { src_offset: offset, dst_offset: at, size: len }];
                unsafe { gpu.device.cmd_copy_buffer(cmd, src.buffer, tmp.buffer, &region) };
            }
            at += len;
        }
        all_to_host(gpu, cmd);
        let v = sub.submit(gpu, timeline, cmd, &[])?;
        timeline.wait(gpu, v, u64::MAX)?;
        let m = tmp.mapped_ref().expect("host buffer");
        let mut out = Vec::with_capacity(ranges.len());
        let mut at = 0usize;
        for r in ranges {
            out.push(m[at..at + r.2 as usize].to_vec());
            at += r.2 as usize;
        }
        Ok(out)
    })();
    gpu.wait_idle()?;
    sub.destroy(gpu);
    alloc.free(gpu, tmp);
    result
}
