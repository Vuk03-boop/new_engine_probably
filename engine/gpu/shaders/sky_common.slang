// Phase 3C: the real-time sky tables in f32 (host oracle: `light::sky`, f64). Included after
// `light_common.slang`; functions take the tables as arguments and declare no resources.
//
// Tables are row-major float4 (rgb) buffers, sampled bilinearly at texel centres with clamped edges,
// exactly as `light::sky::Lut::sample`. Positions are carried as altitude h above the ground (r = R + h)
// so that r^2 - R^2 = h (2R + h) and the horizon quantities never cancel in f32.

static const float R_TOP = 6460000.0;
static const uint TRANS_W = 256u;
static const uint TRANS_H = 64u;
static const uint MS_W = 32u;
static const uint MS_H = 32u;
static const uint VIEW_W = 192u;
static const uint VIEW_H = 108u;
static const uint GROUND_W = 64u;
static const uint VIEW_STEPS = 32u;
// The correction (S-020, host: light::sky_ref): CORR_W x CORR_H texels at the sky-view unit
// coordinates (texel i at i / (n - 1)) per sun elevation slice, CORR_S slices.
static const uint CORR_W = 32u;
static const uint CORR_H = 32u;
static const uint CORR_S = 33u;

float3 lut_sample(StructuredBuffer<float4> t, uint w, uint h, float u, float v)
{
    float fx = clamp(u * float(w) - 0.5, 0.0, float(w - 1u));
    float fy = clamp(v * float(h) - 0.5, 0.0, float(h - 1u));
    uint x0 = uint(floor(fx));
    uint y0 = uint(floor(fy));
    uint x1 = min(x0 + 1u, w - 1u);
    uint y1 = min(y0 + 1u, h - 1u);
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    float3 a = t[y0 * w + x0].xyz;
    float3 b = t[y0 * w + x1].xyz;
    float3 c = t[y1 * w + x0].xyz;
    float3 d = t[y1 * w + x1].xyz;
    return lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
}

float sub_uv(float x, uint n)
{
    return 0.5 / float(n) + x * (1.0 - 1.0 / float(n));
}

float unit_from_sub(float u, uint n)
{
    return clamp((u - 0.5 / float(n)) / (1.0 - 1.0 / float(n)), 0.0, 1.0);
}

// r^2 - R^2 from the altitude.
float rho2(float h)
{
    return h * (2.0 * R_GROUND + h);
}

bool sky_hits_ground(float h, float mu)
{
    float r = R_GROUND + h;
    return mu < 0.0 && r * r * mu * mu - rho2(h) >= 0.0;
}

float sky_ground_distance(float h, float mu)
{
    float r = R_GROUND + h;
    float disc = max(r * r * mu * mu - rho2(h), 0.0);
    // The near root, in the cancellation-free form c / (-b + sqrt(disc)) with c = rho2(h).
    float b = r * mu;
    return max(rho2(h) / (-b + sqrt(disc)), 0.0);
}

float sky_top_distance(float h, float mu)
{
    float r = R_GROUND + h;
    // R_top^2 - r^2 = (R_top - r)(R_top + r), with R_top - r = (R_top - R) - h.
    float disc = r * r * mu * mu + ((R_TOP - R_GROUND) - h) * (R_TOP + r);
    return -r * mu + sqrt(max(disc, 0.0));
}

// Bruneton's transmittance mapping, (h, mu) to (u, v).
float2 trans_uv(float h, float mu)
{
    float hh = sqrt(TOP_SQ_MINUS_GROUND_SQ);
    float rho = sqrt(max(rho2(h), 0.0));
    float d = max(sky_top_distance(h, mu), 0.0);
    float d_min = (R_TOP - R_GROUND) - h;
    float d_max = rho + hh;
    float xm = d_max > d_min ? (d - d_min) / (d_max - d_min) : 0.0;
    return float2(sub_uv(clamp(xm, 0.0, 1.0), TRANS_W), sub_uv(clamp(rho / hh, 0.0, 1.0), TRANS_H));
}

float3 sky_transmittance(StructuredBuffer<float4> trans, float h, float mu)
{
    if (sky_hits_ground(h, mu))
        return float3(0.0, 0.0, 0.0);
    float2 uv = trans_uv(h, mu);
    return lut_sample(trans, TRANS_W, TRANS_H, uv.x, uv.y);
}

float3 sky_ms(StructuredBuffer<float4> ms, float h, float mu_s)
{
    float u = sub_uv(clamp(mu_s * 0.5 + 0.5, 0.0, 1.0), MS_W);
    float v = sub_uv(clamp(h / (R_TOP - R_GROUND), 0.0, 1.0), MS_H);
    return lut_sample(ms, MS_W, MS_H, u, v);
}

float3 sky_ground_irradiance(StructuredBuffer<float4> ground, float mu_s)
{
    return lut_sample(ground, GROUND_W, 1u, sub_uv(clamp(mu_s * 0.5 + 0.5, 0.0, 1.0), GROUND_W), 0.5);
}

// Altitude and sun zenith cosine at distance t along a ray from altitude h with view-zenith cosine mu;
// nu is the view-sun cosine, mu_s the sun zenith cosine at the start.
float2 sky_along(float h, float mu, float mu_s, float nu, float t)
{
    float r = R_GROUND + h;
    float num = t * t + 2.0 * r * mu * t + rho2(h);
    float r_t = sqrt(num + R_GROUND * R_GROUND);
    float h_t = num / (r_t + R_GROUND);
    return float2(h_t, clamp((r * mu_s + t * nu) / r_t, -1.0, 1.0));
}

// In-scattered radiance along a ray (light::sky::SkyLuts::integrate_view).
float3 sky_integrate_view(StructuredBuffer<float4> trans, StructuredBuffer<float4> ms, StructuredBuffer<float4> ground_e, float h, float mu, float mu_s, float nu)
{
    bool ground = sky_hits_ground(h, mu);
    float t_end = ground ? sky_ground_distance(h, mu) : sky_top_distance(h, mu);
    float r = R_GROUND + h;
    float pr = rayleigh_phase(nu);
    float pm = hg_phase(nu, MIE_G);
    float3 tr = float3(1.0, 1.0, 1.0);
    float3 l = float3(0.0, 0.0, 0.0);
    float t0 = 0.0;
    for (uint k = 0u; k < VIEW_STEPS; k++)
    {
        float t1 = step_bound(k + 1u, VIEW_STEPS, t_end, -r * mu);
        float2 at = sky_along(h, mu, mu_s, nu, 0.5 * (t0 + t1));
        Medium m = medium(at.x);
        float3 ts = sky_transmittance(trans, at.x, at.y);
        float3 psi = sky_ms(ms, at.x, at.y);
        float dt = t1 - t0;
        float3 s = (m.rayleigh * pr + m.mie * pm) * ts + (m.rayleigh + m.mie) * psi;
        float3 ex = exp(-m.extinction * dt);
        l += tr * s * (1.0 - ex) / m.extinction;
        tr *= ex;
        t0 = t1;
    }
    if (ground)
    {
        float2 at = sky_along(h, mu, mu_s, nu, t_end);
        float mu_g = at.y;
        float3 ts = sky_transmittance(trans, 0.0, mu_g);
        float3 e = sky_ground_irradiance(ground_e, mu_g);
        l += tr * (GROUND_ALBEDO / PI) * (ts * max(mu_g, 0.0) + e);
    }
    return l;
}

// The observer's horizon: (zenith angle of the grazing ray, angle between horizon and nadir).
float2 sky_horizon()
{
    float h = OBSERVER_ALTITUDE;
    float beta = acos(sqrt(rho2(h)) / (R_GROUND + h));
    return float2(PI - beta, beta);
}

// Unit direction d (+Y up) to sky-view (u, v) for the sun along `sun` (light::sky::view_uv).
float2 sky_view_uv(float3 sun, float3 d)
{
    float2 hz = sky_horizon();
    float zenith = acos(clamp(d.y, -1.0, 1.0));
    float y;
    if (zenith < hz.x)
        y = (1.0 - sqrt(max(1.0 - zenith / hz.x, 0.0))) * 0.5;
    else
        y = 0.5 + 0.5 * sqrt(clamp((zenith - hz.x) / hz.y, 0.0, 1.0));
    float dh = sqrt(d.x * d.x + d.z * d.z);
    float sh = sqrt(sun.x * sun.x + sun.z * sun.z);
    float cos_phi = (dh > 1e-6 && sh > 1e-6) ? clamp((d.x * sun.x + d.z * sun.z) / (dh * sh), -1.0, 1.0) : 1.0;
    float x = sqrt((1.0 - cos_phi) * 0.5);
    return float2(sub_uv(x, VIEW_W), sub_uv(y, VIEW_H));
}

float3 sky_lookup(StructuredBuffer<float4> view, float3 sun, float3 d)
{
    float2 uv = sky_view_uv(sun, d);
    return lut_sample(view, VIEW_W, VIEW_H, uv.x, uv.y);
}

// Fractional correction slice for the sun zenith cosine mu_s (light::sky_ref::slice_coord): -6 to 10
// degrees every degree, then every 5 degrees to 90; clamped.
float corr_slice(float mu_s)
{
    float e = asin(clamp(mu_s, -1.0, 1.0)) * (180.0 / PI);
    float k = e < 10.0 ? e + 6.0 : 16.0 + (e - 10.0) / 5.0;
    return clamp(k, 0.0, float(CORR_S - 1u));
}

// The correction at sky-view unit coordinates (x, y) for mu_s (light::sky::SkyLuts::correction_at):
// bilinear in (x, y), linear between slices.
float3 corr_sample(StructuredBuffer<float4> corr, float x, float y, float mu_s)
{
    float ks = corr_slice(mu_s);
    uint k0 = uint(floor(ks));
    uint k1 = min(k0 + 1u, CORR_S - 1u);
    float tk = ks - float(k0);
    float fx = clamp(x * float(CORR_W - 1u), 0.0, float(CORR_W - 1u));
    float fy = clamp(y * float(CORR_H - 1u), 0.0, float(CORR_H - 1u));
    uint x0 = uint(floor(fx));
    uint y0 = uint(floor(fy));
    uint x1 = min(x0 + 1u, CORR_W - 1u);
    uint y1 = min(y0 + 1u, CORR_H - 1u);
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    float3 s[2];
    uint ks2[2] = { k0, k1 };
    for (uint j = 0u; j < 2u; j++)
    {
        uint base = ks2[j] * CORR_H * CORR_W;
        float3 a = corr[base + y0 * CORR_W + x0].xyz;
        float3 b = corr[base + y0 * CORR_W + x1].xyz;
        float3 c = corr[base + y1 * CORR_W + x0].xyz;
        float3 d = corr[base + y1 * CORR_W + x1].xyz;
        s[j] = lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
    }
    return lerp(s[0], s[1], tk);
}
