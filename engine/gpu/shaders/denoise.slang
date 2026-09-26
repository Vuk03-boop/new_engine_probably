// Phase 3E: native reconstruction, SVGF-style (Schied et al. 2017) on this engine's exact guides.
// Runs after the temporal pass (ADR-0006), in three modes chosen by push constant:
//
// 0 init: per surface pixel, demodulate the accumulated radiance by the material's albedo
//   (illumination = radiance / albedo per channel; 0 where the albedo is below 1e-4) and estimate the
//   variance of that accumulated value: from the history moments (second moment of luminance) when
//   the history is at least `moments_age` samples old (SVGF: 4), else from the 7x7 neighbours on the
//   same surface; divided
//   by the age (the variance of a mean of `age` samples) and by the albedo's luminance squared.
//   Sky pixels are passed through (variance -1 marks them).
// 1 a-trous: one level of the 5x5 B-spline wavelet with holes `step` pixels apart. A tap contributes
//   only if it is on the same surface: the same face, material and integer plane (the guides are
//   exact, so the filter never blurs across an edge, a step or a material boundary), weighted by
//   exp(-|dl| / (sigma_l * sqrt(variance))), the variance pre-blurred 3x3 on the same surface; the
//   variance is filtered with the squared weights.
// 2 remodulate: filtered illumination x albedo back into the radiance buffer (surfaces only; the
//   state in w is kept).
// 3 guide prefilter (before each of the first `prefilter_levels` levels but the first): the 3x3
//   same-surface mean of that level's input luminance. For the first level, init computes it from
//   the neighbours' radiance (flag F_INIT_GUIDE), and the last level remodulates into the radiance
//   buffer itself (flag F_REMODULATE): the pass is memory-bound, so each saved full-frame pass counts. The level's edge-stopping then compares these means (sigma scaled
//   to their variance, var / 9) instead of the taps' own noisy values: weights that depend on the very
//   samples they average favour the common (dark) values of a heavy-tailed 1-spp estimate and lose
//   energy (measured: -4% to -10% at ages 1-4); each tap is now only 1/9 of its own guide.
//   3G: init writes the 3x3 mean only for pixels younger than `prefilter_age`; an older pixel's guide
//   is its own luminance l, stored as -1 - l (`guide_value`, `guide_is_mean`), and its sigma is not
//   scaled: 3x3 means straddle hard lighting edges and blurred converged images there (3G record).
// 4B filter fix (docs/changes/2026-09-26-4b-filter-energy.md; `DenoiseSettings::weights`, both flags off
//   is the filter above bit for bit):
//   F_SYMMETRIC: sigma from the pixel's plus the tap's variance (each unblurred, each divided by 9 when its
//     guide is a 3x3 mean), so a pair weighs the same both ways; the variance pre-blur is skipped.
//   F_CONSERVATIVE: symmetric weights w, and exchanges instead of a weighted mean:
//     out_p = x_p + sum_q c_pq (x_q - x_p),  c_pq = w_pq / max(W_p, W_q),
//     W the kernel's sum over a pixel's same-surface taps at the level's step (guides only, so borders
//     are smoothed like the interior), rounded up to 1/64. c is the same at both ends of a pair, so two
//     pixels exchange equal and opposite amounts and each surface keeps its energy; W >= the taps' sum
//     keeps out_p a convex mix. Init packs W for the first two levels into the compact key's free bits
//     (the material is 16 bits: key.x bits 0-18 are face and material, 19-24 level 0, 25-30 level 1);
//     later levels use W = 1. Surface tests and the albedo lookup mask them off (KEY_MASK).

struct Params
{
    uint width;
    uint height;
    uint mode;
    uint step;
    uint src;     // a-trous: 0 reads A and writes B, 1 the reverse; remodulate: the buffer to read
    uint guide;   // which guide / history buffer holds the current frame (History parity)
    uint flags;
    float sigma_l;
    uint moments_age;
    uint prefilter_age;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> radiance;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> guides_0;
[[vk::binding(2, 0)]] StructuredBuffer<uint4> guides_1;
[[vk::binding(3, 0)]] StructuredBuffer<float4> hist_0;
[[vk::binding(4, 0)]] StructuredBuffer<float4> hist_1;
[[vk::binding(5, 0)]] StructuredBuffer<float4> albedo;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> filt_a;
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> filt_b;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> lum_guide;
// The same-surface key (guide .x and .y) copied by init: the levels read 8 bytes per tap, not 16.
[[vk::binding(9, 0)]] RWStructuredBuffer<uint2> keys;
[[vk::push_constant]] ConstantBuffer<Params> params;

static const uint F_PREFILTER = 1u;
static const uint F_INIT_GUIDE = 2u;
static const uint F_REMODULATE = 4u;
static const uint F_NO_VARIANCE_BLUR = 8u;
static const uint F_SYMMETRIC = 16u;
static const uint F_CONSERVATIVE = 32u;
// The compact key's face and material bits; above them init packs the border sums (F_CONSERVATIVE).
static const uint KEY_MASK = 0x7FFFFu;
static const float H[5] = { 1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };
static const uint F_FAULT_IGNORE_GUIDES = 256u;
static const uint FACE_NONE = 7u;

float luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

uint4 key_at(uint i)
{
    uint2 k = keys[i];
    return uint4(k, 0u, 0u);
}

uint4 guide_at(uint i)
{
    return params.guide == 0u ? guides_0[i] : guides_1[i];
}

float4 hist_at(uint i)
{
    return params.guide == 0u ? hist_0[i] : hist_1[i];
}

float3 albedo_of(uint4 g)
{
    uint m = (g.x & KEY_MASK) >> 3;
    uint count, stride;
    albedo.GetDimensions(count, stride);
    return m < count ? albedo[m].xyz : float3(1.0, 1.0, 1.0);
}

bool same_surface(uint4 a, uint4 b)
{
    if ((params.flags & F_FAULT_IGNORE_GUIDES) != 0u)
        return (b.x & 7u) != FACE_NONE;
    return (a.x & KEY_MASK) == (b.x & KEY_MASK) && a.y == b.y;
}

float3 demodulate(float3 c, float3 a)
{
    return float3(a.x > 1e-4 ? c.x / a.x : 0.0, a.y > 1e-4 ? c.y / a.y : 0.0, a.z > 1e-4 ? c.z / a.z : 0.0);
}

// The first level's luminance guide: a 3x3 mean (>= 0) or a pixel's own value l (stored -1 - l).
float guide_value(float g)
{
    return g >= 0.0 ? g : -1.0 - g;
}

bool guide_is_mean(float g)
{
    return g >= 0.0;
}

float4 read_filt(uint which, uint i)
{
    return which == 0u ? filt_a[i] : filt_b[i];
}

// F_CONSERVATIVE: the kernel's sum over pixel p's same-surface taps at `step` (full guides; init),
// rounded up to 1/64 and stored as 0-63 (the sum is at least H0^2 = 0.14, so 9-64 / 64).
uint border_code(int2 p, uint4 g, int step)
{
    float t = 0.0;
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            int2 q = p + int2(dx, dy) * step;
            if (q.x < 0 || q.y < 0 || q.x >= int(params.width) || q.y >= int(params.height))
                continue;
            if (same_surface(g, guide_at(uint(q.y) * params.width + uint(q.x))))
                t += H[dx + 2] * H[dy + 2];
        }
    }
    return uint(min(ceil(t * 64.0), 64.0)) - 1u;
}

// The border sum a level reads from a compact key (1 from the third level on).
float border_of(uint kx, uint level)
{
    return level >= 2u ? 1.0 : float(((kx >> (19u + 6u * level)) & 63u) + 1u) / 64.0;
}

void write_filt(uint which, uint i, float4 v)
{
    if (which == 0u)
        filt_a[i] = v;
    else
        filt_b[i] = v;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= params.width || tid.y >= params.height)
        return;
    uint i = tid.y * params.width + tid.x;
    // Init and remodulate read the full guide; the levels and the prefilter read the compact key.
    uint4 g = (params.mode == 0u || params.mode == 2u) ? guide_at(i) : key_at(i);
    bool sky = (g.x & 7u) == FACE_NONE;

    if (params.mode == 0u)
    {
        float4 r = radiance[i];
        keys[i] = g.xy;
        if (!sky && (params.flags & F_CONSERVATIVE) != 0u)
            keys[i] = uint2(g.x | (border_code(int2(tid.xy), g, 1) << 19) | (border_code(int2(tid.xy), g, 2) << 25), g.y);
        if (sky)
        {
            filt_a[i] = float4(r.xyz, -1.0);
            if ((params.flags & F_INIT_GUIDE) != 0u)
                lum_guide[i] = 0.0;
            return;
        }
        float3 a = albedo_of(g);
        float3 illum = demodulate(r.xyz, a);
        uint age = max(uint(r.w) & 0xFFFFu, 1u);
        if ((params.flags & F_INIT_GUIDE) != 0u && age >= params.prefilter_age)
            lum_guide[i] = -1.0 - luminance(illum);
        else if ((params.flags & F_INIT_GUIDE) != 0u)
        {
            float gs = 0.0, gn = 0.0;
            for (int gy = -1; gy <= 1; gy++)
            {
                for (int gx = -1; gx <= 1; gx++)
                {
                    int2 q = int2(tid.xy) + int2(gx, gy);
                    if (q.x < 0 || q.y < 0 || q.x >= int(params.width) || q.y >= int(params.height))
                        continue;
                    uint qi = uint(q.y) * params.width + uint(q.x);
                    uint4 gq = guide_at(qi);
                    if (!same_surface(g, gq))
                        continue;
                    gs += luminance(demodulate(radiance[qi].xyz, albedo_of(gq)));
                    gn += 1.0;
                }
            }
            lum_guide[i] = gs / gn;
        }
        float la = max(luminance(a), 1e-4);
        float var;
        if (age >= max(params.moments_age, 1u))
        {
            float l = luminance(r.xyz);
            var = max(hist_at(i).w - l * l, 0.0);
        }
        else
        {
            // Spatial estimate of the sample variance from the same surface's neighbours.
            float s1 = 0.0, s2 = 0.0, n = 0.0;
            for (int dy = -3; dy <= 3; dy++)
            {
                for (int dx = -3; dx <= 3; dx++)
                {
                    int2 q = int2(tid.xy) + int2(dx, dy);
                    if (q.x < 0 || q.y < 0 || q.x >= int(params.width) || q.y >= int(params.height))
                        continue;
                    uint qi = uint(q.y) * params.width + uint(q.x);
                    if (!same_surface(g, guide_at(qi)))
                        continue;
                    float l = luminance(radiance[qi].xyz);
                    s1 += l;
                    s2 += l * l;
                    n += 1.0;
                }
            }
            s1 /= n;
            var = max(s2 / n - s1 * s1, 0.0) * float(age);
        }
        filt_a[i] = float4(illum, var / (float(age) * la * la));
        return;
    }

    if (params.mode == 3u)
    {
        if (sky)
        {
            lum_guide[i] = 0.0;
            return;
        }
        float s = 0.0, n = 0.0;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int2 q = int2(tid.xy) + int2(dx, dy);
                if (q.x < 0 || q.y < 0 || q.x >= int(params.width) || q.y >= int(params.height))
                    continue;
                uint qi = uint(q.y) * params.width + uint(q.x);
                if (!same_surface(g, key_at(qi)))
                    continue;
                s += luminance(read_filt(params.src, qi).xyz);
                n += 1.0;
            }
        }
        lum_guide[i] = s / n;
        return;
    }

    if (params.mode == 1u)
    {
        uint dst = 1u - params.src;
        float4 c = read_filt(params.src, i);
        bool last = (params.flags & F_REMODULATE) != 0u;
        if (sky)
        {
            if (!last)
                write_filt(dst, i, c);
            return;
        }
        bool pre = (params.flags & F_PREFILTER) != 0u;
        float gp = pre ? lum_guide[i] : 0.0;
        float lp = pre ? guide_value(gp) : luminance(c.xyz);
        bool mean_guide = pre && guide_is_mean(gp);
        bool cons = (params.flags & F_CONSERVATIVE) != 0u;
        bool sym = cons || (params.flags & F_SYMMETRIC) != 0u;
        // Symmetric: the pixel's own variance in its guide's units (a tap adds its own).
        float vp = c.w * (mean_guide ? 1.0 / 9.0 : 1.0);
        // The edge-stopping variance: a 3x3 Gaussian of the variance on the same surface (SVGF).
        float gv = c.w, gw = 1.0;
        if (!sym && (params.flags & F_NO_VARIANCE_BLUR) == 0u)
        {
        gv = 0.0;
        gw = 0.0;
        for (int vy = -1; vy <= 1; vy++)
        {
            for (int vx = -1; vx <= 1; vx++)
            {
                int2 q = int2(tid.xy) + int2(vx, vy);
                if (q.x < 0 || q.y < 0 || q.x >= int(params.width) || q.y >= int(params.height))
                    continue;
                uint qi = uint(q.y) * params.width + uint(q.x);
                if (!same_surface(g, key_at(qi)))
                    continue;
                float k = (vx == 0 ? 0.5 : 0.25) * (vy == 0 ? 0.5 : 0.25);
                gv += k * read_filt(params.src, qi).w;
                gw += k;
            }
        }
        }
        float sigma = params.sigma_l * sqrt(max(gv / gw * (mean_guide ? 1.0 / 9.0 : 1.0), 1e-12)) + 1e-12;
        float3 sum = float3(0.0, 0.0, 0.0);
        float vsum = 0.0;
        float wsum = 0.0;
        // Conservative: the exchange sum_q c (x_q - x), its coefficients' sum; the level's border sums.
        float3 ex = float3(0.0, 0.0, 0.0);
        float csum = 0.0;
        uint level = firstbitlow(params.step);
        float wp = border_of(g.x, level);
        for (int dy = -2; dy <= 2; dy++)
        {
            for (int dx = -2; dx <= 2; dx++)
            {
                int2 q = int2(tid.xy) + int2(dx, dy) * int(params.step);
                if (q.x < 0 || q.y < 0 || q.x >= int(params.width) || q.y >= int(params.height))
                    continue;
                uint qi = uint(q.y) * params.width + uint(q.x);
                uint4 kq = key_at(qi);
                if (!same_surface(g, kq))
                    continue;
                float4 cq = read_filt(params.src, qi);
                float gq = pre ? lum_guide[qi] : 0.0;
                float lq = pre ? guide_value(gq) : luminance(cq.xyz);
                float s = sigma;
                if (sym)
                    s = params.sigma_l * sqrt(max(vp + cq.w * (pre && guide_is_mean(gq) ? 1.0 / 9.0 : 1.0), 1e-12)) + 1e-12;
                float w = H[dx + 2] * H[dy + 2] * exp(-abs(lq - lp) / s);
                if (cons)
                {
                    if (dx != 0 || dy != 0)
                    {
                        float cpq = w / max(wp, border_of(kq.x, level));
                        ex += cpq * (cq.xyz - c.xyz);
                        vsum += cpq * cpq * cq.w;
                        csum += cpq;
                    }
                    continue;
                }
                sum += w * cq.xyz;
                vsum += w * w * cq.w;
                wsum += w;
            }
        }
        float3 o;
        float v;
        if (cons)
        {
            // The pixel keeps 1 - csum of its own value.
            o = c.xyz + ex;
            v = vsum + (1.0 - csum) * (1.0 - csum) * c.w;
        }
        else
        {
            o = sum / wsum;
            v = vsum / (wsum * wsum);
        }
        if (last)
            radiance[i] = float4(o * albedo_of(g), radiance[i].w);
        else
            write_filt(dst, i, float4(o, v));
        return;
    }

    // Remodulate.
    if (sky)
        return;
    float4 r = radiance[i];
    float3 illum = read_filt(params.src, i).xyz;
    radiance[i] = float4(illum * albedo_of(g), r.w);
}
