//! Device memory behind the 1D `Ledger` (ADR-0003).
//!
//! - Memory comes in **blocks** (`vkAllocateMemory`), one pool per (category, memory type). Each
//!   block holds one ledger grant for its full size, so the ledger's *reserved* bytes are exactly
//!   what the driver was asked for, and *live* is the sum of sub-allocations.
//! - A block is only allocated after the ledger grants it. A refusal allocates nothing and
//!   changes nothing but the ledger's refusal counter.
//! - Sub-allocation is first-fit over an ordered free list with coalescing. An empty block is
//!   freed at once. A request larger than the block size gets a block of its own.
//! - Why blocks: region size "1 brick" alone needs ~5,000 buffers on the street block, above the
//!   usual `maxMemoryAllocationCount` of 4,096.
//! - Images (2C: the G-buffer) use the same blocks and ledger, but never share a block with
//!   buffers: pools are keyed by resource class too, so `bufferImageGranularity` never applies.
//!
//! Freeing is immediate: callers must first make sure the GPU no longer uses the memory
//! (see [`crate::timeline::Retirement`]).

use std::collections::BTreeMap;

use ash::vk;
use memory::{Category, Grant, Ledger};

use crate::context::{Gpu, GpuError, Result, VkCheck};

pub const DEFAULT_BLOCK_BYTES: u64 = 64 << 20;

/// Where a buffer lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// `DEVICE_LOCAL`.
    Device,
    /// `HOST_VISIBLE | HOST_COHERENT`, persistently mapped: staging, readback, small test buffers.
    Host,
}

struct Block {
    memory: vk::DeviceMemory,
    size: u64,
    category: Category,
    memory_type: u32,
    images: bool,
    mapped: *mut u8,
    grant: Grant,
    /// offset → size
    free: BTreeMap<u64, u64>,
    used: u64,
}

/// A buffer bound to a sub-allocation. Not `Clone`: free it exactly once with [`Allocator::free`].
#[derive(Debug)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    pub address: vk::DeviceAddress,
    pub category: Category,
    block: usize,
    offset: u64,
    alloc_size: u64,
    mapped: *mut u8,
}

impl Buffer {
    /// Mapped bytes of a [`Kind::Host`] buffer, for writing. The GPU must not be using them.
    pub fn mapped(&mut self) -> Option<&mut [u8]> {
        (!self.mapped.is_null()).then(|| unsafe { std::slice::from_raw_parts_mut(self.mapped, self.size as usize) })
    }

    /// Mapped bytes of a [`Kind::Host`] buffer, for reading after the GPU work that wrote them completed.
    pub fn mapped_ref(&self) -> Option<&[u8]> {
        (!self.mapped.is_null()).then(|| unsafe { std::slice::from_raw_parts(self.mapped, self.size as usize) })
    }

    /// (block index, offset, size) of the backing range, for tests and diagnostics.
    pub fn range(&self) -> (usize, u64, u64) {
        (self.block, self.offset, self.alloc_size)
    }
}

/// An optimal-tiling 2D image with one view, bound to a sub-allocation. Free it with [`Allocator::free_image`].
#[derive(Debug)]
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub aspect: vk::ImageAspectFlags,
    pub category: Category,
    block: usize,
    offset: u64,
    alloc_size: u64,
}

impl Image {
    /// Bytes of device memory backing the image (the driver's requirement, not texels x size).
    pub fn alloc_size(&self) -> u64 {
        self.alloc_size
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    pub blocks_live: usize,
    pub vk_allocations: u64,
    pub vk_frees: u64,
    pub buffers_live: u64,
    pub images_live: u64,
}

pub struct Allocator {
    ledger: Ledger,
    blocks: Vec<Option<Block>>,
    block_bytes: u64,
    stats: AllocStats,
}

impl Allocator {
    pub fn new(budget: memory::Budget) -> Self {
        Self::with_block_bytes(budget, DEFAULT_BLOCK_BYTES)
    }

    pub fn with_block_bytes(budget: memory::Budget, block_bytes: u64) -> Self {
        Self { ledger: Ledger::new(budget), blocks: Vec::new(), block_bytes, stats: AllocStats::default() }
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    pub fn stats(&self) -> AllocStats {
        self.stats
    }

    fn memory_type(gpu: &Gpu, bits: u32, kind: Kind) -> Result<u32> {
        let want = match kind {
            Kind::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            Kind::Host => vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        };
        // Prefer a type without the other side's flag, so staging stays out of the small
        // device-local host-visible (BAR) heap and device buffers out of host memory.
        let avoid = match kind {
            Kind::Device => vk::MemoryPropertyFlags::HOST_VISIBLE,
            Kind::Host => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        let m = &gpu.info.memory;
        let ok = |i: u32| bits & (1 << i) != 0 && m.memory_types[i as usize].property_flags.contains(want);
        (0..m.memory_type_count)
            .find(|&i| ok(i) && !m.memory_types[i as usize].property_flags.intersects(avoid))
            .or_else(|| (0..m.memory_type_count).find(|&i| ok(i)))
            .ok_or_else(|| GpuError::NoDevice(format!("no memory type for {kind:?} in bits {bits:#x}")))
    }

    /// Creates a buffer. On refusal (`GpuError::OverBudget`) nothing is allocated.
    pub fn create_buffer(&mut self, gpu: &Gpu, size: u64, usage: vk::BufferUsageFlags, category: Category, kind: Kind) -> Result<Buffer> {
        let dev = &gpu.device;
        let info = vk::BufferCreateInfo::default().size(size.max(1)).usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let buffer = unsafe { dev.create_buffer(&info, None) }.vk("vkCreateBuffer")?;
        let req = unsafe { dev.get_buffer_memory_requirements(buffer) };
        let placed = Self::memory_type(gpu, req.memory_type_bits, kind).and_then(|t| self.place(gpu, category, t, kind, false, req.size, req.alignment));
        let (block, offset) = match placed {
            Ok(p) => p,
            Err(e) => {
                unsafe { dev.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };
        let b = self.blocks[block].as_ref().expect("placed block");
        if let Err(e) = unsafe { dev.bind_buffer_memory(buffer, b.memory, offset) }.vk("vkBindBufferMemory") {
            self.unplace(gpu, block, offset, req.size);
            unsafe { dev.destroy_buffer(buffer, None) };
            return Err(e);
        }
        let mapped = if b.mapped.is_null() { std::ptr::null_mut() } else { unsafe { b.mapped.add(offset as usize) } };
        let address = unsafe { dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) };
        self.stats.buffers_live += 1;
        Ok(Buffer { buffer, size, address, category, block, offset, alloc_size: req.size, mapped })
    }

    /// Creates a device-local, optimal-tiling 2D image (one mip, one layer) and a view of it. On
    /// refusal (`GpuError::OverBudget`) nothing is allocated.
    pub fn create_image(&mut self, gpu: &Gpu, format: vk::Format, extent: vk::Extent2D, usage: vk::ImageUsageFlags, aspect: vk::ImageAspectFlags, category: Category) -> Result<Image> {
        let dev = &gpu.device;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { dev.create_image(&info, None) }.vk("vkCreateImage")?;
        let req = unsafe { dev.get_image_memory_requirements(image) };
        let placed = Self::memory_type(gpu, req.memory_type_bits, Kind::Device).and_then(|t| self.place(gpu, category, t, Kind::Device, true, req.size, req.alignment));
        let (block, offset) = match placed {
            Ok(p) => p,
            Err(e) => {
                unsafe { dev.destroy_image(image, None) };
                return Err(e);
            }
        };
        let memory = self.blocks[block].as_ref().expect("placed block").memory;
        let bound = unsafe { dev.bind_image_memory(image, memory, offset) }.vk("vkBindImageMemory").and_then(|()| {
            let vi = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            unsafe { dev.create_image_view(&vi, None) }.vk("vkCreateImageView")
        });
        let view = match bound {
            Ok(v) => v,
            Err(e) => {
                self.unplace(gpu, block, offset, req.size);
                unsafe { dev.destroy_image(image, None) };
                return Err(e);
            }
        };
        self.stats.images_live += 1;
        Ok(Image { image, view, format, extent, aspect, category, block, offset, alloc_size: req.size })
    }

    /// Destroys the image and its view and returns its range. The GPU must be done with it.
    pub fn free_image(&mut self, gpu: &Gpu, i: Image) {
        unsafe {
            gpu.device.destroy_image_view(i.view, None);
            gpu.device.destroy_image(i.image, None);
        }
        self.stats.images_live -= 1;
        self.unplace(gpu, i.block, i.offset, i.alloc_size);
    }

    /// Destroys the buffer and returns its range. The GPU must be done with it.
    pub fn free(&mut self, gpu: &Gpu, b: Buffer) {
        unsafe { gpu.device.destroy_buffer(b.buffer, None) };
        self.stats.buffers_live -= 1;
        self.unplace(gpu, b.block, b.offset, b.alloc_size);
    }

    #[allow(clippy::too_many_arguments)]
    fn place(&mut self, gpu: &Gpu, category: Category, memory_type: u32, kind: Kind, images: bool, size: u64, align: u64) -> Result<(usize, u64)> {
        for (i, slot) in self.blocks.iter_mut().enumerate() {
            let Some(b) = slot else { continue };
            if b.category != category || b.memory_type != memory_type || b.images != images {
                continue;
            }
            if let Some(off) = take_first_fit(&mut b.free, size, align) {
                b.used += size;
                self.ledger.set_live(&b.grant, b.used).expect("live within block");
                return Ok((i, off));
            }
        }
        // New block, sized for the request if it is large.
        let block_size = self.block_bytes.max(size.next_multiple_of(1 << 20));
        let grant = self.ledger.reserve(category, block_size).map_err(GpuError::OverBudget)?;
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let ai = vk::MemoryAllocateInfo::default().allocation_size(block_size).memory_type_index(memory_type).push_next(&mut flags);
        let memory = match unsafe { gpu.device.allocate_memory(&ai, None) }.vk("vkAllocateMemory") {
            Ok(m) => m,
            Err(e) => {
                self.ledger.release(grant).expect("own grant");
                return Err(e);
            }
        };
        self.stats.vk_allocations += 1;
        let mapped = if kind == Kind::Host {
            match unsafe { gpu.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }.vk("vkMapMemory") {
                Ok(p) => p as *mut u8,
                Err(e) => {
                    unsafe { gpu.device.free_memory(memory, None) };
                    self.stats.vk_frees += 1;
                    self.ledger.release(grant).expect("own grant");
                    return Err(e);
                }
            }
        } else {
            std::ptr::null_mut()
        };
        let mut free = BTreeMap::new();
        free.insert(0, block_size);
        let off = take_first_fit(&mut free, size, align).expect("fresh block fits");
        self.ledger.set_live(&grant, size).expect("live within block");
        let block = Block { memory, size: block_size, category, memory_type, images, mapped, grant, free, used: size };
        let i = match self.blocks.iter().position(Option::is_none) {
            Some(i) => {
                self.blocks[i] = Some(block);
                i
            }
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        };
        self.stats.blocks_live += 1;
        Ok((i, off))
    }

    fn unplace(&mut self, gpu: &Gpu, block: usize, offset: u64, size: u64) {
        let b = self.blocks[block].as_mut().expect("live block");
        give_back(&mut b.free, offset, size);
        b.used -= size;
        if b.used == 0 {
            let b = self.blocks[block].take().expect("live block");
            debug_assert_eq!(b.free.get(&0), Some(&b.size), "an empty block is one free range");
            unsafe { gpu.device.free_memory(b.memory, None) };
            self.stats.vk_frees += 1;
            self.stats.blocks_live -= 1;
            self.ledger.release(b.grant).expect("own grant");
        } else {
            self.ledger.set_live(&b.grant, b.used).expect("live within block");
        }
    }

    /// Frees every block. Call only when no buffer or image is outstanding; returns the number leaked.
    pub fn destroy(mut self, gpu: &Gpu) -> u64 {
        let leaked = self.stats.buffers_live + self.stats.images_live;
        for b in self.blocks.drain(..).flatten() {
            unsafe { gpu.device.free_memory(b.memory, None) };
            let _ = self.ledger.release(b.grant);
        }
        leaked
    }
}

/// First-fit: takes an aligned range of `size` from the free list.
fn take_first_fit(free: &mut BTreeMap<u64, u64>, size: u64, align: u64) -> Option<u64> {
    let (start, len, at) = free.iter().find_map(|(&start, &len)| {
        let at = start.next_multiple_of(align.max(1));
        (at + size <= start + len).then_some((start, len, at))
    })?;
    free.remove(&start);
    if at > start {
        free.insert(start, at - start);
    }
    if at + size < start + len {
        free.insert(at + size, start + len - (at + size));
    }
    Some(at)
}

/// Returns a range to the free list, merging with its neighbours.
fn give_back(free: &mut BTreeMap<u64, u64>, mut offset: u64, mut size: u64) {
    if let Some((&s, &l)) = free.range(..offset).next_back() {
        debug_assert!(s + l <= offset, "double free or overlap");
        if s + l == offset {
            free.remove(&s);
            offset = s;
            size += l;
        }
    }
    if let Some(&l) = free.get(&(offset + size)) {
        free.remove(&(offset + size));
        size += l;
    }
    free.insert(offset, size);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_fit_aligns_splits_and_coalesces() {
        let mut f = BTreeMap::from([(0u64, 1000u64)]);
        assert_eq!(take_first_fit(&mut f, 100, 1), Some(0));
        assert_eq!(take_first_fit(&mut f, 100, 256), Some(256));
        assert_eq!(f, BTreeMap::from([(100, 156), (356, 644)]));
        assert_eq!(take_first_fit(&mut f, 150, 1), Some(100), "fills the alignment gap");
        assert_eq!(f, BTreeMap::from([(250, 6), (356, 644)]));
        assert_eq!(take_first_fit(&mut f, 700, 1), None);
        give_back(&mut f, 0, 100);
        assert_eq!(f, BTreeMap::from([(0, 100), (250, 6), (356, 644)]));
        give_back(&mut f, 256, 100);
        assert_eq!(f, BTreeMap::from([(0, 100), (250, 750)]), "merges with both neighbours");
        give_back(&mut f, 100, 150);
        assert_eq!(f, BTreeMap::from([(0, 1000)]), "everything back in one range");
    }
}
