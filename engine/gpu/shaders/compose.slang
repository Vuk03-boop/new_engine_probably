// Phase 4B: emission after reconstruction, and the exposure meter (docs/changes/2026-09-26-phase4b-many-lights.md).
// One thread per pixel, after the temporal pass and the filter (or after the shade pass when they are
// off), only while the lights are on.
//
// - EMISSION: adds the per-material emitted radiance (the emitter table's `emission`, as the
//   reference's EMISSION term) to every surface pixel's shown radiance; w (age and reason) is kept.
//   Emission is deterministic and never accumulated, demodulated or filtered (ADR-0005 Amendment 3).
// - METER: per 16x16 workgroup, the sum of the shown luminance Y over all pixels, and the sum of ln Y
//   with a count over the pixels with Y > 0 and Y >= `floor` (light::exposure::Meter), written to
//   `partials[group]` as (sum Y, sum ln Y, count, 0). The host adds the groups in f64.
// - Planted fault METER_WITHOUT_EMISSION: the meter reads the radiance before emission is added.

struct Params
{
    uint width;
    uint height;
    uint flags;
    float floor;
    uint groups_x;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> radiance;
[[vk::binding(1, 0)]] [[vk::image_format("unknown")]] Texture2D<float> depth_tex;
[[vk::binding(2, 0)]] [[vk::image_format("unknown")]] Texture2D<uint> material_tex;
[[vk::binding(3, 0)]] StructuredBuffer<float4> emission;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> partials;
[[vk::push_constant]] ConstantBuffer<Params> params;

static const uint F_EMISSION = 1u;
static const uint F_METER = 2u;
static const uint F_FAULT_METER_WITHOUT_EMISSION = 256u;

groupshared float3 sums[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID, uint3 gid: SV_GroupID, uint gi: SV_GroupIndex)
{
    float3 acc = float3(0.0, 0.0, 0.0);
    if (tid.x < params.width && tid.y < params.height)
    {
        uint i = tid.y * params.width + tid.x;
        int3 px = int3(int(tid.x), int(tid.y), 0);
        float4 r = radiance[i];
        float3 shown = r.xyz;
        if ((params.flags & F_EMISSION) != 0u && depth_tex.Load(px) > 0.0)
        {
            uint m = material_tex.Load(px);
            uint count, stride;
            emission.GetDimensions(count, stride);
            if (m < count)
            {
                shown += emission[m].xyz;
                radiance[i] = float4(shown, r.w);
            }
        }
        float3 metered = (params.flags & F_FAULT_METER_WITHOUT_EMISSION) != 0u ? r.xyz : shown;
        float y = dot(metered, float3(0.2126, 0.7152, 0.0722));
        acc.x = y;
        if (y > 0.0 && y >= params.floor)
        {
            acc.y = log(y);
            acc.z = 1.0;
        }
    }
    sums[gi] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128u; s > 0u; s >>= 1u)
    {
        if (gi < s)
            sums[gi] += sums[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0u && (params.flags & F_METER) != 0u)
        partials[gid.y * params.groups_x + gid.x] = float4(sums[0], 0.0);
}
