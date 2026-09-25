// S-020: bakes the reference sky for the sky correction (host: light::sky_ref). Thread t works on job
// t % jobs (a direction and a sun, made on the host) as replica r = t / jobs; each dispatch writes, per
// thread, the sum and the sum of squares of `samples` Monte Carlo sky samples (the reference's
// estimator, `sky_sample`), which the host adds up in f64. Replicas keep the GPU full when a slice
// has few jobs. Sample s of job i draws from the stream pcg_hash(g + pcg_hash(s + pcg_hash(seed))),
// with g = base + i the job's global index (its texel) and s = first + r * samples + k.

#include "light_common.slang"

struct Params
{
    uint count;   // threads: jobs x replicas
    uint jobs;
    uint first;   // index of this dispatch's first sample
    uint samples; // samples per thread in this dispatch
    uint seed;
    uint steps;
    uint sun_steps;
    uint base;    // global index of job 0 (the stream key)
};

struct Job
{
    float4 dir; // xyz unit view direction (+Y up)
    float4 sun; // xyz unit vector toward the sun
};

struct Acc
{
    float4 sum;
    float4 sum_sq;
};

[[vk::binding(0, 0)]] StructuredBuffer<Job> jobs;
[[vk::binding(1, 0)]] RWStructuredBuffer<Acc> acc;
[[vk::push_constant]] ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint t = tid.x;
    if (t >= params.count)
        return;
    uint i = t % params.jobs;
    uint first = params.first + (t / params.jobs) * params.samples;
    Job j = jobs[i];
    float3 s = float3(0.0, 0.0, 0.0);
    float3 s2 = float3(0.0, 0.0, 0.0);
    for (uint k = 0u; k < params.samples; k++)
    {
        uint rng = pcg_hash(params.base + i + pcg_hash(first + k + pcg_hash(params.seed)));
        float3 l = sky_sample(j.dir.xyz, j.sun.xyz, rng, params.steps, params.sun_steps);
        s += l;
        s2 += l * l;
    }
    Acc a;
    a.sum = float4(s, 0.0);
    a.sum_sq = float4(s2, 0.0);
    acc[t] = a;
}
