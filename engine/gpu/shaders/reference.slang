// Phase 3A: the GPU reference path tracer (ADR-0005). The same estimator as `light::reference` and
// `light::atmosphere` (CPU, f64), drawing random numbers in the same order from the same PCG32 streams;
// the GPU tests compare the two statistically. One sample per pixel per dispatch, accumulated.
//
// - Primary ray through the pixel centre (as `ray_primary.slang`), TMin = near. A miss sees the sky
//   (flag SKY) and the sun disk (flag SUN).
// - At each surface vertex: the sun by next-event estimation (uniform point on the disk, or its centre
//   with POINT_SUN; one shadow ray), then one cosine-sampled continuation. An escaping continuation
//   takes the sky radiance; a hit continues while bounces remain.
// - Secondary origins: the hit point snapped onto its (integer) voxel face plane, offset 1/256 voxel
//   along the face normal. Rays are opaque and cull back faces.
// - Sky: a Monte Carlo volumetric path through the discretised atmosphere, all orders, sun disk
//   excluded. Atmosphere positions are `q`, meters relative to the planet-surface point under the
//   observer (+Y up), so |p|^2 - R^2 = q.q + 2 R q.y is formed without cancellation in f32.
// - Output: per pixel, the sum and the sum of squares of the samples; sum.w counts samples that hit a
//   bad surface id (must stay 0). With RESET the dispatch overwrites instead of adding.
// - PROBE_T writes, instead, the transmittance from the observer along the primary direction (the
//   host/shader constant-agreement check).

struct Params
{
    float4 eye;        // xyz eye (voxels); w: cos of the sun's angular radius
    float4 forward;    // xyz; w: near
    float4 right;      // right * tan_half_x
    float4 up;         // up * tan_half_y
    float4 sun_dir;    // xyz unit vector toward the sun
    float4 sun_ground; // rgb E_SUN * T_obs; w: 1 / sun solid angle
    uint width;
    uint height;
    uint frame;
    uint seed;
    uint flags;
    uint max_bounces;
    uint steps;
    uint sun_steps;
};

struct RegionRef
{
    uint2* quads;
    uint* tri_quad;
    uint quad_count;
    uint tri_count;
};

struct Acc
{
    float4 sum;
    float4 sum_sq;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] StructuredBuffer<RegionRef> regions;
[[vk::binding(2, 0)]] StructuredBuffer<float4> albedo;
[[vk::binding(3, 0)]] RWStructuredBuffer<Acc> acc;
[[vk::push_constant]] ConstantBuffer<Params> params;

static const uint F_SUN = 1u;
static const uint F_SKY = 2u;
static const uint F_POINT_SUN = 4u;
static const uint F_UNIFORM_SKY = 8u;
static const uint F_ALBEDO_ONE = 16u;
static const uint F_RESET = 32u;
static const uint F_PROBE_T = 64u;
static const uint F_FAULT_PDF = 256u;
static const uint F_FAULT_NO_COS = 512u;

#include "light_common.slang"

// ---- scene ----

// 0 = miss, 1 = hit, 2 = bad surface id.
uint trace_surface(float3 origin, float3 dir, float tmin, out float3 point, out float3 normal, out float3 alb)
{
    point = float3(0.0, 0.0, 0.0);
    normal = float3(0.0, 1.0, 0.0);
    alb = float3(0.0, 0.0, 0.0);
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = tmin;
    ray.TMax = 1.0e30;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> rq;
    rq.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    rq.Proceed();
    if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return 0u;
    uint region = rq.CommittedInstanceID();
    uint prim = rq.CommittedPrimitiveIndex();
    uint count, stride;
    regions.GetDimensions(count, stride);
    if (region >= count)
        return 2u;
    RegionRef r = regions[region];
    uint quad = prim < r.tri_count ? r.tri_quad[prim] : 0xFFFFFFFFu;
    if (quad >= r.quad_count)
        return 2u;
    uint2 qbits = r.quads[quad];
    uint face = (qbits.x >> 16) & 7u;
    uint material = qbits.x & 0xFFFFu;
    uint mcount, mstride;
    albedo.GetDimensions(mcount, mstride);
    if (material >= mcount || face >= 6u)
        return 2u;
    uint axis = face / 2u;
    float sgn = (face & 1u) != 0u ? 1.0 : -1.0;
    normal = float3(0.0, 0.0, 0.0);
    normal[axis] = sgn;
    point = origin + dir * rq.CommittedRayT();
    point[axis] = round(point[axis]) + sgn * RAY_OFFSET;
    alb = (params.flags & F_ALBEDO_ONE) != 0u ? float3(1.0, 1.0, 1.0) : albedo[material].xyz;
    return 1u;
}

bool occluded(float3 origin, float3 dir)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = 1.0e30;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
    rq.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    rq.Proceed();
    return rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

float3 sky(float3 dir, inout uint rng)
{
    if ((params.flags & F_UNIFORM_SKY) != 0u)
        return float3(1.0, 1.0, 1.0);
    if ((params.flags & F_SKY) != 0u)
        return sky_sample(dir, params.sun_dir.xyz, rng, params.steps, params.sun_steps);
    return float3(0.0, 0.0, 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= params.width || tid.y >= params.height)
        return;
    uint i = tid.y * params.width + tid.x;
    float sx = 2.0 * (float(tid.x) + 0.5) / float(params.width) - 1.0;
    float sy = 1.0 - 2.0 * (float(tid.y) + 0.5) / float(params.height);
    float3 d = params.forward.xyz + sx * params.right.xyz + sy * params.up.xyz;

    if ((params.flags & F_PROBE_T) != 0u)
    {
        Acc a;
        a.sum = float4(transmittance(float3(0.0, OBSERVER_ALTITUDE, 0.0), normalize(d), params.steps), 0.0);
        a.sum_sq = float4(0.0, 0.0, 0.0, 0.0);
        acc[i] = a;
        return;
    }

    uint rng = pcg_hash(i + pcg_hash(params.frame + pcg_hash(params.seed)));
    float3 sun = params.sun_dir.xyz;
    bool sun_on = (params.flags & F_SUN) != 0u && (params.flags & F_UNIFORM_SKY) == 0u;
    float3 l = float3(0.0, 0.0, 0.0);
    float bad = 0.0;

    float3 p, n, rho;
    uint h = trace_surface(params.eye.xyz, d, params.forward.w, p, n, rho);
    if (h == 0u)
    {
        float3 dn = normalize(d);
        l = sky(dn, rng);
        if (sun_on && dot(dn, sun) >= params.eye.w)
            l += params.sun_ground.xyz * params.sun_ground.w;
    }
    else if (h == 2u)
    {
        bad = 1.0;
    }
    else
    {
        float3 beta = float3(1.0, 1.0, 1.0);
        for (uint bounce = 0u; bounce <= params.max_bounces; bounce++)
        {
            float u1 = rnd(rng);
            float u2 = rnd(rng);
            if (sun_on)
            {
                float3 ws = (params.flags & F_POINT_SUN) != 0u ? sun : uniform_cone(sun, params.eye.w, u1, u2);
                float c = dot(n, ws);
                if (c > 0.0 && !occluded(p, ws))
                {
                    float k = (params.flags & F_FAULT_NO_COS) != 0u ? 1.0 : c;
                    l += beta * rho * params.sun_ground.xyz * (k / PI);
                }
            }
            float v1 = rnd(rng);
            float3 wi = cosine_hemisphere(n, v1, rnd(rng));
            beta *= rho;
            if ((params.flags & F_FAULT_PDF) != 0u)
                beta *= 1.1;
            float3 p2, n2, rho2;
            uint h2 = trace_surface(p, wi, 0.0, p2, n2, rho2);
            if (h2 == 0u)
            {
                l += beta * sky(wi, rng);
                break;
            }
            if (h2 == 2u)
            {
                bad = 1.0;
                break;
            }
            if (bounce == params.max_bounces)
                break;
            p = p2;
            n = n2;
            rho = rho2;
        }
    }

    Acc a;
    if ((params.flags & F_RESET) != 0u)
    {
        a.sum = float4(l, bad);
        a.sum_sq = float4(l * l, 0.0);
    }
    else
    {
        a = acc[i];
        a.sum += float4(l, bad);
        a.sum_sq += float4(l * l, 0.0);
    }
    acc[i] = a;
}
