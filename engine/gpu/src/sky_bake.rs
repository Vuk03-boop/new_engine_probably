//! S-020: bakes the reference sky that corrects the sky-view table (`shaders/sky_bake.slang`; the
//! data and its file are `light::sky_ref`). A tool, not a frame pass: the `sky_bake` binary runs it
//! once and writes the file.
//!
//! The jobs (one per `light::sky_ref` texel: a view direction and a sun) are made on the host. Each
//! job runs as a power-of-two number of replicas, so a slice's 1,024 jobs still fill the GPU; each
//! thread adds `per_dispatch` Monte Carlo sky samples per dispatch, written as f32 sums. The host adds
//! threads and dispatches in f64 and forms the mean and its standard error.

use std::mem::size_of;
use std::time::Instant;

use ash::vk;
use light::atmosphere::SkyOptions;
use light::sample::V3;
use memory::Category;

use crate::alloc::{Allocator, Kind};
use crate::context::{Gpu, GpuError, Result, VkCheck};
use crate::reflect::{self, Field, Param};
use crate::staging::{download, Uploader};
use crate::submit::Submitter;
use crate::timeline::Timeline;

pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sky_bake.spv"));
pub const REFLECTION: &str = include_str!(concat!(env!("OUT_DIR"), "/sky_bake.json"));

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub count: u32,
    pub jobs: u32,
    pub first: u32,
    pub samples: u32,
    pub seed: u32,
    pub steps: u32,
    pub sun_steps: u32,
    /// Global index of job 0, the key of its random stream.
    pub base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Job {
    pub dir: [f32; 4],
    pub sun: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Acc {
    pub sum: [f32; 4],
    pub sum_sq: [f32; 4],
}

macro_rules! field {
    ($t:ty, $f:ident) => {
        Field { name: stringify!($f), offset: std::mem::offset_of!($t, $f) as u64, size: std::mem::size_of_val(&<$t>::default().$f) as u64 }
    };
}

pub fn host_layout() -> Vec<Param> {
    vec![
        Param::Descriptor { name: "jobs", binding: 0, element: vec![field!(Job, dir), field!(Job, sun)] },
        Param::Descriptor { name: "acc", binding: 1, element: vec![field!(Acc, sum), field!(Acc, sum_sq)] },
        Param::PushConstants {
            name: "params",
            fields: vec![
                field!(Params, count),
                field!(Params, jobs),
                field!(Params, first),
                field!(Params, samples),
                field!(Params, seed),
                field!(Params, steps),
                field!(Params, sun_steps),
                field!(Params, base),
            ],
        },
    ]
}

/// The bake's result: per job, the mean and the standard error of the mean.
pub struct Baked {
    pub mean: Vec<V3>,
    pub se: Vec<V3>,
    pub seconds: f64,
}

/// Threads a dispatch aims for (the RTX 3050 laptop is full well below this).
const THREADS: usize = 32_768;

/// Replicas per job for `jobs` jobs: a power of two giving at least [`THREADS`] threads.
pub fn replicas(jobs: usize) -> u32 {
    (THREADS.div_ceil(jobs.max(1))).next_power_of_two() as u32
}

/// Bakes `samples` Monte Carlo sky samples for each (direction, sun) job; job i draws from the
/// stream of global index `base + i`. `samples` must be a multiple of `per_dispatch` × [`replicas`]. `progress` is called after each dispatch with the
/// samples per job done so far.
#[allow(clippy::too_many_arguments)]
pub fn bake(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, jobs: &[(V3, V3)], base: u32, o: &SkyOptions, seed: u32, samples: u32, per_dispatch: u32, mut progress: impl FnMut(u32)) -> Result<Baked> {
    let n = jobs.len();
    let reps = replicas(n);
    let step = per_dispatch * reps;
    assert!(per_dispatch > 0 && samples.is_multiple_of(step), "samples must be a multiple of per_dispatch x replicas ({step})");
    reflect::check(REFLECTION, &host_layout()).map_err(GpuError::Layout)?;
    let threads = n * reps as usize;
    let dev = &gpu.device;
    let job_bytes: Vec<u8> = jobs
        .iter()
        .flat_map(|(d, s)| [d[0], d[1], d[2], 0.0, s[0], s[1], s[2], 0.0])
        .flat_map(|x| (x as f32).to_le_bytes())
        .collect();
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
    let job_buf = alloc.create_buffer(gpu, job_bytes.len() as u64, usage, Category::GpuTemporal, Kind::Device)?;
    let acc_len = (threads * size_of::<Acc>()) as u64;
    let acc_buf = match alloc.create_buffer(gpu, acc_len, usage, Category::GpuTemporal, Kind::Device) {
        Ok(b) => b,
        Err(e) => {
            alloc.free(gpu, job_buf);
            return Err(e);
        }
    };

    let mut objects: Option<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline, vk::DescriptorPool)> = None;
    let result = (|| -> Result<Baked> {
        let v = {
            up.upload(gpu, timeline, &job_buf, 0, &job_bytes)?;
            up.flush(gpu, timeline)?
        };
        timeline.wait(gpu, v, u64::MAX)?;
        let b = |i: u32| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
        let bindings = [b(0), b(1)];
        let set_layout = unsafe { dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.vk("vkCreateDescriptorSetLayout")?;
        let pc = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(size_of::<Params>() as u32)];
        let sl = [set_layout];
        let layout = unsafe { dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&pc), None) }.vk("vkCreatePipelineLayout")?;
        let (chunks, rest) = SPIRV.as_chunks::<4>();
        assert!(rest.is_empty(), "SPIR-V is whole words");
        let words: Vec<u32> = chunks.iter().map(|&c| u32::from_le_bytes(c)).collect();
        let module = unsafe { dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }.vk("vkCreateShaderModule")?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let pipeline = unsafe { dev.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
        unsafe { dev.destroy_shader_module(module, None) };
        let pipeline = pipeline.map_err(|(_, e)| GpuError::Vk { call: "vkCreateComputePipelines", result: e })?[0];
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 }];
        let pool = unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None) }.vk("vkCreateDescriptorPool")?;
        objects = Some((set_layout, layout, pipeline, pool));
        let set = unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sl)) }.vk("vkAllocateDescriptorSets")?[0];
        let infos = [job_buf.buffer, acc_buf.buffer].map(|buffer| [vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE }]);
        let writes: Vec<_> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
        unsafe { dev.update_descriptor_sets(&writes, &[]) };

        let mut sum = vec![[0.0f64; 3]; n];
        let mut sum_sq = vec![[0.0f64; 3]; n];
        let mut sub = Submitter::new(gpu)?;
        let t = Instant::now();
        let run = (|| -> Result<()> {
            let mut first = 0;
            while first < samples {
                let p = Params { count: threads as u32, jobs: n as u32, first, samples: per_dispatch, seed, steps: o.steps, sun_steps: o.sun_steps, base };
                let bytes = unsafe { std::slice::from_raw_parts(&p as *const Params as *const u8, size_of::<Params>()) };
                let cmd = sub.begin(gpu, timeline)?;
                // The previous download's read of `acc` before this dispatch overwrites it; this
                // dispatch's writes before the download's copy.
                let before = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)];
                let after = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
                unsafe {
                    dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&before));
                    dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                    dev.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
                    dev.cmd_dispatch(cmd, (threads as u32).div_ceil(64), 1, 1);
                    dev.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&after));
                }
                let v = sub.submit(gpu, timeline, cmd, &[])?;
                timeline.wait(gpu, v, u64::MAX)?;
                let data = download(gpu, alloc, timeline, &[(&acc_buf, 0, acc_len)])?.remove(0);
                for (t, c) in data.as_chunks::<32>().0.iter().enumerate() {
                    let f = |k: usize| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()) as f64;
                    for ch in 0..3 {
                        sum[t % n][ch] += f(ch);
                        sum_sq[t % n][ch] += f(4 + ch);
                    }
                }
                first += step;
                progress(first);
            }
            Ok(())
        })();
        gpu.wait_idle()?;
        sub.destroy(gpu);
        run?;
        let seconds = t.elapsed().as_secs_f64();
        let s = samples as f64;
        let mean: Vec<V3> = sum.iter().map(|x| x.map(|v| v / s)).collect();
        let se = sum_sq
            .iter()
            .zip(&mean)
            .map(|(q, m)| [0, 1, 2].map(|c| ((q[c] / s - m[c] * m[c]).max(0.0) / (s - 1.0)).sqrt()))
            .collect();
        Ok(Baked { mean, se, seconds })
    })();
    if let Some((set_layout, layout, pipeline, pool)) = objects {
        unsafe {
            dev.destroy_descriptor_pool(pool, None);
            dev.destroy_pipeline(pipeline, None);
            dev.destroy_pipeline_layout(layout, None);
            dev.destroy_descriptor_set_layout(set_layout, None);
        }
    }
    alloc.free(gpu, job_buf);
    alloc.free(gpu, acc_buf);
    result
}
