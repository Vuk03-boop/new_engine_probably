// Phase 3B: real-time lighting from the ADR-0003 G-buffer (ADR-0005 conventions). One pixel per
// thread; writes linear HDR radiance per pixel for the "light" view to tone-map.
//
// 3B term: the sun by next-event estimation, exactly as `reference.slang` does at the primary vertex:
// - the surface point is rebuilt from depth (reverse-Z: t = near / depth along the pixel-centre ray,
//   whose forward component is 1), then snapped onto its voxel face plane and offset 1/256 voxel
//   along the face normal;
// - the pixel's PCG32 stream is the reference's for the same frame and seed, and its first two numbers
//   pick the point on the sun disk (or the centre with POINT_SUN), so frame f of this pass equals the
//   reference's sun-only sample of frame f up to the surface position;
// - one shadow ray, opaque, back faces culled.
// 3C term: sky light. The pixel stream's next two numbers pick a cosine-distributed direction (as the
// reference's continuation ray at the primary vertex); if that ray escapes, the sky radiance of the
// sky-view table in that direction, times the albedo. A pixel with no surface shows the table's sky
// plus the sun disk (radiance E_SUN * T_obs / sun solid angle). UNIFORM_SKY replaces the sky by 1 and
// turns the sun off, as the reference's control does, so the visibility and cosine logic can be
// compared per pixel exactly.
// 3F term (BOUNCE): one bounce of diffuse light. The sky ray above becomes the reference's
// continuation ray: when it hits a surface instead of escaping, that point is lit as the reference
// lights its second vertex (max_bounces 1): the sun by next-event estimation with its own shadow ray,
// and the sky through one more cosine-sampled visibility ray, both times the two albedos. The hit's
// face, material and snapped point come from the ray query and the region table (as in
// `reference.slang`). The stream continues in the reference's order (sun pair, then direction pair).
// Light from bounces >= 2 is not computed (reported against the reference, 3F record).
// 4B (EMITTERS, the lit module; ADR-0005 Amendment 3): build.rs compiles this file twice. Without
// EMITTERS it is the M3 module, byte for byte; with it, binding 8 holds the scene's emitter rows and
// binding 9 the per-material emission (used only by a planted fault). At the primary vertex, and at
// the bounce hit, `pad1` (s) emitter samples are drawn after the sun's two numbers and before the
// continuation's, each exactly the reference's `emitter_sample` (alias selection by power, a point
// uniform in the quad's solid angle or by area below SOLID_ANGLE_MIN, one visibility segment), and
// averaged. With s = 1 frame f equals the reference's one-bounce sample f with EMITTERS_DIRECT and
// EMITTERS_INDIRECT, emission off. `pad0` is the emitter count; none are drawn when it is 0.
// Emission is not added here: gpu::compose adds it after reconstruction.
// Output w: 1 when the pixel's or the bounce hit's surface id is outside its table (must stay 0),
// else 0.

#include "light_common.slang"
#include "sky_common.slang"
#ifdef EMITTERS
#include "emitters_common.slang"
#endif

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
    uint pad0;
    uint pad1;
    uint pad2;
};

struct RegionRef
{
    uint2* quads;
    uint* tri_quad;
    uint quad_count;
    uint tri_count;
};

[[vk::binding(0, 0)]] [[vk::image_format("unknown")]] Texture2D<float> depth_tex;
[[vk::binding(1, 0)]] [[vk::image_format("unknown")]] Texture2D<float2> normal_tex;
[[vk::binding(2, 0)]] [[vk::image_format("unknown")]] Texture2D<uint> material_tex;
[[vk::binding(3, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(4, 0)]] StructuredBuffer<float4> albedo;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> radiance;
[[vk::binding(6, 0)]] StructuredBuffer<float4> sky_view;
[[vk::binding(7, 0)]] StructuredBuffer<RegionRef> regions;
[[vk::push_constant]] ConstantBuffer<Params> params;
#ifdef EMITTERS
[[vk::binding(8, 0)]] StructuredBuffer<Emitter> emitters;
[[vk::binding(9, 0)]] StructuredBuffer<float4> emission;
#endif

static const uint F_SUN = 1u;
static const uint F_SKY = 2u;
static const uint F_POINT_SUN = 4u;
static const uint F_UNIFORM_SKY = 8u;
static const uint F_BOUNCE = 16u;
static const uint F_FAULT_FLIP_X = 256u;
static const uint F_FAULT_DEPTH = 512u;
#ifdef EMITTERS
static const uint F_EMITTERS = 32u;
static const uint F_FAULT_EMITTER_AFTER = 1024u;
static const uint F_FAULT_NO_BOUNCE_EMITTERS = 2048u;
static const uint F_FAULT_EMISSION_IN_SHADE = 4096u;
static const uint F_FAULT_EMITTER_SUM = 8192u;
#endif

// The axis-aligned face normal from the octahedral target: the dominant axis, with its sign.
float3 face_normal(float2 e)
{
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    float3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z)
        return float3(n.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
    if (a.y >= a.z)
        return float3(0.0, n.y >= 0.0 ? 1.0 : -1.0, 0.0);
    return float3(0.0, 0.0, n.z >= 0.0 ? 1.0 : -1.0);
}

float3 sky(float3 dir)
{
    if ((params.flags & F_UNIFORM_SKY) != 0u)
        return float3(1.0, 1.0, 1.0);
    if ((params.flags & F_SKY) != 0u)
        return sky_lookup(sky_view, params.sun_dir.xyz, dir);
    return float3(0.0, 0.0, 0.0);
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

// The closest surface along a ray: 0 = miss, 1 = hit, 2 = bad surface id (as `reference.slang`).
uint trace_surface(float3 origin, float3 dir, out float3 point, out float3 normal, out float3 alb)
{
    point = float3(0.0, 0.0, 0.0);
    normal = float3(0.0, 1.0, 0.0);
    alb = float3(0.0, 0.0, 0.0);
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
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
    alb = albedo[material].xyz;
    return 1u;
}

#ifdef EMITTERS
// The segment origin -> origin + seg (TMax 1 along the unnormalized direction).
bool occluded_segment(float3 origin, float3 seg)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = seg;
    ray.TMin = 0.0;
    ray.TMax = 1.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
    rq.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    rq.Proceed();
    return rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

// One emitter sample at (p, n), without albedo: the reference's `emitter_sample` without its faults.
float3 emitter_sample(float3 p, float3 n, inout uint rng)
{
    uint i = uniform_index(params.pad0, rng);
    Emitter e = emitters[i];
    if ((next_u32(rng) >> 8u) >= e.meta.x)
        e = emitters[e.meta.y];
    float u = rnd(rng);
    float v = rnd(rng);
    float3 ne = face_normal(e.meta.z);
    if (dot(ne, p - e.p0_area.xyz) <= 0.0)
        return float3(0.0, 0.0, 0.0);
    SphRect r = sph_rect(e.p0_area.xyz, e.eu_pdf.xyz, e.ev.xyz, p);
    bool solid = r.solid_angle >= SOLID_ANGLE_MIN;
    float3 y = solid ? sph_rect_sample(r, u, v) : e.p0_area.xyz + u * e.eu_pdf.xyz + v * e.ev.xyz;
    float3 d = y - p;
    float d2 = dot(d, d);
    float3 w = d * rsqrt(d2);
    float cx = dot(n, w);
    float cy = -dot(ne, w);
    if (cx <= 0.0 || cy <= 0.0)
        return float3(0.0, 0.0, 0.0);
    if (occluded_segment(p, y + ne * RAY_OFFSET - p))
        return float3(0.0, 0.0, 0.0);
    float k = solid ? r.solid_angle : e.p0_area.w * cy / d2;
    return e.radiance.xyz * (cx * k / (e.eu_pdf.w * PI));
}

// The mean of s = pad1 emitter samples (their sum with the planted EMITTER_SUM).
float3 emitter_light(float3 p, float3 n, inout uint rng)
{
    uint s = max(params.pad1, 1u);
    float3 sum = float3(0.0, 0.0, 0.0);
    for (uint k = 0u; k < s; k++)
        sum += emitter_sample(p, n, rng);
    return (params.flags & F_FAULT_EMITTER_SUM) != 0u ? sum : sum / float(s);
}
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= params.width || tid.y >= params.height)
        return;
    uint i = tid.y * params.width + tid.x;
    int3 px = int3(int(tid.x), int(tid.y), 0);
    float sx = 2.0 * (float(tid.x) + 0.5) / float(params.width) - 1.0;
    float sy = 1.0 - 2.0 * (float(tid.y) + 0.5) / float(params.height);
    float3 d = params.forward.xyz + sx * params.right.xyz + sy * params.up.xyz;
    float3 sun = params.sun_dir.xyz;
    bool sun_on = (params.flags & F_SUN) != 0u && (params.flags & F_UNIFORM_SKY) == 0u;
    bool sky_on = (params.flags & (F_SKY | F_UNIFORM_SKY)) != 0u;
    float depth = depth_tex.Load(px);
    float3 l = float3(0.0, 0.0, 0.0);
    float bad = 0.0;
    if (depth <= 0.0)
    {
        float3 dn = normalize(d);
        l = sky(dn);
        if (sun_on && dot(dn, sun) >= params.eye.w)
            l += params.sun_ground.xyz * params.sun_ground.w;
        radiance[i] = float4(l, 0.0);
        return;
    }
    uint material = material_tex.Load(px);
    uint mcount, mstride;
    albedo.GetDimensions(mcount, mstride);
    if (material >= mcount)
    {
        radiance[i] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float t = params.forward.w / depth;
    if ((params.flags & F_FAULT_DEPTH) != 0u)
        t *= 1.01;
    float3 n = face_normal(normal_tex.Load(px));
    if ((params.flags & F_FAULT_FLIP_X) != 0u)
        n.x = -n.x;
    float3 p = params.eye.xyz + d * t;
    uint axis = abs(n.x) > 0.5 ? 0u : (abs(n.y) > 0.5 ? 1u : 2u);
    p[axis] = round(p[axis]) + n[axis] * RAY_OFFSET;
    float3 rho = albedo[material].xyz;

    uint rng = pcg_hash(i + pcg_hash(params.frame + pcg_hash(params.seed)));
    float u1 = rnd(rng);
    float u2 = rnd(rng);
    if (sun_on)
    {
        float3 ws = (params.flags & F_POINT_SUN) != 0u ? sun : uniform_cone(sun, params.eye.w, u1, u2);
        float c = dot(n, ws);
        if (c > 0.0 && !occluded(p, ws))
            l = rho * params.sun_ground.xyz * (c / PI);
    }
#ifdef EMITTERS
    bool em_on = (params.flags & F_EMITTERS) != 0u && params.pad0 > 0u;
    bool em_after = (params.flags & F_FAULT_EMITTER_AFTER) != 0u;
    if ((params.flags & F_FAULT_EMISSION_IN_SHADE) != 0u)
        l += emission[material].xyz;
    if (em_on && !em_after)
        l += rho * emitter_light(p, n, rng);
#endif
    float v1 = rnd(rng);
    float v2 = rnd(rng);
    float3 wi = cosine_hemisphere(n, v1, v2);
#ifdef EMITTERS
    if (em_on && em_after)
        l += rho * emitter_light(p, n, rng);
#endif
    if ((params.flags & F_BOUNCE) != 0u)
    {
        float3 p2, n2, rho2;
        uint h = trace_surface(p, wi, p2, n2, rho2);
        if (h == 0u)
        {
            l += rho * sky(wi);
        }
        else if (h == 2u)
        {
            bad = 1.0;
        }
        else
        {
            float3 beta = rho * rho2;
            float w1 = rnd(rng);
            float w2 = rnd(rng);
            if (sun_on)
            {
                float3 ws = (params.flags & F_POINT_SUN) != 0u ? sun : uniform_cone(sun, params.eye.w, w1, w2);
                float c = dot(n2, ws);
                if (c > 0.0 && !occluded(p2, ws))
                    l += beta * params.sun_ground.xyz * (c / PI);
            }
#ifdef EMITTERS
            if (em_on && (params.flags & F_FAULT_NO_BOUNCE_EMITTERS) == 0u)
                l += beta * emitter_light(p2, n2, rng);
#endif
            float x1 = rnd(rng);
            float x2 = rnd(rng);
            if (sky_on)
            {
                float3 wj = cosine_hemisphere(n2, x1, x2);
                if (!occluded(p2, wj))
                    l += beta * sky(wj);
            }
        }
    }
    else if (sky_on && !occluded(p, wi))
    {
        l += rho * sky(wi);
    }
    radiance[i] = float4(l, bad);
}
