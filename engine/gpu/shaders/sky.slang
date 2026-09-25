// Phase 3C: builds the sky-view table for the current sun (host oracle: light::sky::SkyLuts::view).
// One thread per texel. The transmittance, multiple-scattering and ground-irradiance tables are built
// once on the host (f64) and uploaded; this pass depends on the sun and is rerun when it moves.
// Each texel is multiplied by the correction (S-020; exactly 1 when uncorrected).

#include "light_common.slang"
#include "sky_common.slang"

struct Params
{
    float4 sun; // xyz unit vector toward the sun; w unused
};

[[vk::binding(0, 0)]] StructuredBuffer<float4> trans;
[[vk::binding(1, 0)]] StructuredBuffer<float4> ms;
[[vk::binding(2, 0)]] StructuredBuffer<float4> ground_e;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> view;
[[vk::binding(4, 0)]] StructuredBuffer<float4> corr;
[[vk::push_constant]] ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= VIEW_W || tid.y >= VIEW_H)
        return;
    float x = unit_from_sub((float(tid.x) + 0.5) / float(VIEW_W), VIEW_W);
    float y = unit_from_sub((float(tid.y) + 0.5) / float(VIEW_H), VIEW_H);
    float2 hz = sky_horizon();
    float zenith;
    if (y < 0.5)
    {
        float c = 1.0 - 2.0 * y;
        zenith = (1.0 - c * c) * hz.x;
    }
    else
    {
        float c = 2.0 * y - 1.0;
        zenith = hz.x + c * c * hz.y;
    }
    float cos_phi = -(x * x * 2.0 - 1.0);
    float mu = cos(zenith);
    float mu_s = params.sun.y;
    float sin_v = sqrt(max(1.0 - mu * mu, 0.0));
    float nu = clamp(mu * mu_s + sin_v * sqrt(max(1.0 - mu_s * mu_s, 0.0)) * cos_phi, -1.0, 1.0);
    float3 l = sky_integrate_view(trans, ms, ground_e, OBSERVER_ALTITUDE, mu, mu_s, nu);
    l *= corr_sample(corr, x, y, mu_s);
    view[tid.y * VIEW_W + tid.x] = float4(l, 0.0);
}
