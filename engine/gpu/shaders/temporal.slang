// Phase 3D: the temporal foundation (ADR-0006). One thread per pixel, after the shade pass.
//
// - Current guides from the G-buffer: face (axis and sign, from the octahedral normal), the face's
//   integer plane coordinate (the depth-rebuilt point snapped along the face axis), material, and the
//   pixel's region key and snapshot (region table, by the surface id's region index).
// - Reprojection: the point through the previous camera, to a continuous previous pixel position;
//   motion = previous position - current position (pixels).
// - History: bilinear over the four previous pixels around it; each tap is valid when its guide has
//   the same face, plane and material, and (same region key) the same region snapshot. Valid taps'
//   weights are renormalised; history colour and age are blended with them.
// - Accept: age = min(age_prev + 1, max_age), colour = prev + (x - prev) / age. Reject: age 1, x.
//   Every visible pixel takes its fresh sample x (ADR-0006 "fresh production").
// - 3E (ADR-0006 Amendment 1): when the bilinear taps accept, the history colour and moment are
//   resampled with Catmull-Rom over the 4x4 previous pixels instead, if every tap with a nonzero
//   weight is valid, clamped to the range of the 2x2 bilinear taps (no ringing, never negative).
//   Repeated bilinear resampling blurs a moving camera's history by about a pixel per few frames;
//   Catmull-Rom keeps it sharp. A snapped still pixel has weights (0, 1, 0, 0): unchanged. Any
//   invalid tap, or the `bilinear` setting, keeps the bilinear result. Reasons and age are the
//   bilinear taps' in both cases.
// - 3E: the history's w holds the same running mean of the fresh sample's squared luminance (the
//   second moment the denoiser's variance estimate needs). Resampling interpolates each tap's
//   variance (w - lum(xyz)^2), not w, and rebuilds w from the resampled colour: interpolating w
//   itself adds the taps' spread of means to the variance every frame of motion (at a sharp shadow
//   edge about lum^2), which disables the filter's luminance edge-stopping there.
// - 3E relight (ADR-0006 Amendment 1): an edit changes light outside its rebuilt regions. For each
//   box the host lists this frame (the edited voxels, 1 voxel margin), a pixel whose history would be
//   accepted is instead rejected as "relit" when the ray from its surface point towards the sun
//   crosses the box (its sun shadow may have changed) or the point is within the box's sky radius of
//   it (its sky visibility may have changed by more than about 2%).
// - 4B relight for lights (ADR-0006 Amendment 2), only in the lit module (`-D EMITTERS`; without it this
//   is the M3 module): after the 3E rule, a pixel whose history would still be accepted is rejected as
//   "relit by a light" (reason 10) when some light row both matters there (the pixel is in front of
//   the emitter, and the unoccluded irradiance bound Y(L_e) * min(2 pi, A / d^2) over pi is at least 2%
//   of the history's luminance) and can change there (the emitter changed, or the segment from the
//   pixel to the emitter's centre crosses the row's edit box grown by the emitter's radius).
// - Outputs: next guides, next history colour and state, motion, and, in place in `radiance`, the
//   resolved colour (xyz) and the state (w) for the light and debug views. The state buffers hold
//   age (bits 0-15) | reason (bits 16-23) | bad material (bit 24); in `radiance.w` the same number is
//   stored as a float value (exact below 2^24), not as bits, so no denormal flush can touch it.

struct Params
{
    float4 eye;        // current eye (voxels); w: near
    float4 forward;    // current forward; w: unused
    float4 right;      // current right * tan_half_x; w: width (uint bits)
    float4 up;         // current up * tan_half_y; w: height (uint bits)
    float4 prev_eye;   // previous eye; w: previous near
    float4 prev_fwd;   // previous forward; w: flags (uint bits)
    float4 prev_right; // previous right / tan_half_x; w: max_age (uint bits)
    float4 prev_up;    // previous up / tan_half_y; w: unused
};

[[vk::binding(0, 0)]] [[vk::image_format("unknown")]] Texture2D<float> depth_tex;
[[vk::binding(1, 0)]] [[vk::image_format("unknown")]] Texture2D<float2> normal_tex;
[[vk::binding(2, 0)]] [[vk::image_format("unknown")]] Texture2D<uint> material_tex;
[[vk::binding(3, 0)]] [[vk::image_format("unknown")]] Texture2D<uint2> surface_tex;
// Per region index: xyz region key, w snapshot (debug_view::Tables).
[[vk::binding(4, 0)]] StructuredBuffer<int4> regions;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> radiance;
[[vk::binding(6, 0)]] StructuredBuffer<uint4> prev_guides;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint4> next_guides;
[[vk::binding(8, 0)]] StructuredBuffer<float4> prev_hist;
[[vk::binding(9, 0)]] RWStructuredBuffer<float4> next_hist;
[[vk::binding(10, 0)]] StructuredBuffer<uint> prev_state;
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> next_state;
[[vk::binding(12, 0)]] RWStructuredBuffer<float2> motion;
// [0]: sun direction (xyz), box count (w); then per box: (lo.xyz, sky radius), (hi.xyz, 0). Voxels.
// 4B (lit module): at [LIGHT_HEAD] the light row count (x), then per row (gpu::temporal::LightRow):
// (centre.xyz, radius), (normal.xyz or 0 for both sides, Y), (Y * A, edit box index or -1, 0, 0).
[[vk::binding(13, 0)]] StructuredBuffer<float4> relight;
[[vk::push_constant]] ConstantBuffer<Params> params;

static const uint F_RESET = 1u;
static const uint F_BILINEAR = 2u;
static const uint F_FAULT_RESET_ALWAYS = 256u;
static const uint F_FAULT_NEVER_REJECT = 512u;
static const uint F_FAULT_NO_FRESH = 1024u;

static const uint R_ACCEPTED = 0u;
static const uint R_SKY = 1u;
static const uint R_OFFSCREEN = 2u;
static const uint R_NO_PREV = 3u;
static const uint R_MATERIAL = 4u;
static const uint R_NORMAL = 5u;
static const uint R_DISOCCLUDED = 6u;
static const uint R_EDITED = 7u;
static const uint R_RESET = 8u;
static const uint R_RELIT = 9u;
#ifdef EMITTERS
static const uint R_RELIT_LIGHT = 10u;
// 1 + 2 * gpu::temporal::MAX_RELIGHT_BOXES.
static const uint LIGHT_HEAD = 33u;
static const float LIGHT_FRACTION = 0.02;
static const float TWO_PI = 6.28318530717958647692;
static const float INV_PI = 0.31830988618379067154;
#endif

static const uint FACE_NONE = 7u;
static const float SNAP = 1.0e-3;

uint face_of(float2 e)
{
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    float3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z)
        return n.x >= 0.0 ? 1u : 0u;
    if (a.y >= a.z)
        return n.y >= 0.0 ? 3u : 2u;
    return n.z >= 0.0 ? 5u : 4u;
}

uint pack_key(int3 k)
{
    return (uint(k.x) & 1023u) | ((uint(k.y) & 1023u) << 10) | ((uint(k.z) & 1023u) << 20);
}

uint state_bits(uint age, uint reason, uint bad)
{
    return min(age, 65535u) | (reason << 16) | (bad << 24);
}

// Validates one previous tap against the current guide; returns the reason (R_ACCEPTED if valid).
uint check_tap(uint4 cur, uint4 prev, bool never_reject)
{
    uint prev_face = prev.x & 7u;
    if (prev_face == FACE_NONE)
        return R_NO_PREV;
    if (never_reject)
        return R_ACCEPTED;
    if (prev.z == cur.z && prev.w != cur.w)
        return R_EDITED;
    if ((prev.x >> 3) != (cur.x >> 3))
        return R_MATERIAL;
    if (prev_face != (cur.x & 7u))
        return R_NORMAL;
    if (prev.y != cur.y)
        return R_DISOCCLUDED;
    return R_ACCEPTED;
}

// A history texel with its w as the variance of its samples' luminance (see the header).
float4 with_variance(float4 h)
{
    float l = dot(h.xyz, float3(0.2126, 0.7152, 0.0722));
    return float4(h.xyz, max(h.w - l * l, 0.0));
}

// Catmull-Rom weights of the four taps at offsets -1, 0, 1, 2 for a fraction t in [0, 1].
float4 catmull_rom(float t)
{
    float t2 = t * t;
    float t3 = t2 * t;
    return float4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

// Whether the light at surface point p may have changed because of the edited boxes.
bool relit(float3 p)
{
    float4 head = relight[0];
    uint n = uint(head.w);
    float3 sun = head.xyz;
    float3 inv = float3(1.0 / sun.x, 1.0 / sun.y, 1.0 / sun.z);
    for (uint b = 0u; b < n; b++)
    {
        float4 lo = relight[1u + 2u * b];
        float3 hi = relight[2u + 2u * b].xyz;
        // Sky: distance from p to the box.
        float3 d = max(max(lo.xyz - p, p - hi), float3(0.0, 0.0, 0.0));
        if (dot(d, d) <= lo.w * lo.w)
            return true;
        // Sun: the ray p + t sun, t >= 0, against the box (slabs).
        float3 t0 = (lo.xyz - p) * inv;
        float3 t1 = (hi - p) * inv;
        float3 tn = min(t0, t1);
        float3 tf = max(t0, t1);
        float enter = max(max(tn.x, tn.y), max(tn.z, 0.0));
        float leave = min(tf.x, min(tf.y, tf.z));
        if (enter <= leave)
            return true;
    }
    return false;
}

#ifdef EMITTERS
// Whether a light row's light at surface point p, with history luminance yh, may have changed.
bool relit_by_light(float3 p, float yh)
{
    uint n = uint(relight[LIGHT_HEAD].x);
    for (uint k = 0u; k < n; k++)
    {
        float4 a = relight[LIGHT_HEAD + 1u + 3u * k];
        float4 b = relight[LIGHT_HEAD + 2u + 3u * k];
        float4 c = relight[LIGHT_HEAD + 3u + 3u * k];
        float3 v = p - a.xyz;
        // In front of a one-sided emitter.
        if (dot(b.xyz, b.xyz) > 0.0 && dot(v, b.xyz) <= 0.0)
            continue;
        // The unoccluded irradiance bound over pi against the history (albedo 1).
        float d = max(sqrt(dot(v, v)) - a.w, 0.0);
        float bound = d > 0.0 ? min(TWO_PI * b.w, c.x / (d * d)) : TWO_PI * b.w;
        if (bound * INV_PI < LIGHT_FRACTION * yh)
            continue;
        int bx = int(c.y);
        if (bx < 0)
            return true;
        // The segment p -> centre against the edit box grown by the radius (slabs, t in [0, 1]).
        float3 lo = relight[1u + 2u * uint(bx)].xyz - a.w;
        float3 hi = relight[2u + 2u * uint(bx)].xyz + a.w;
        float3 s = a.xyz - p;
        float enter = 0.0;
        float leave = 1.0;
        for (uint ax = 0u; ax < 3u; ax++)
        {
            if (s[ax] == 0.0)
            {
                if (p[ax] < lo[ax] || p[ax] > hi[ax])
                    leave = -1.0;
            }
            else
            {
                float t0 = (lo[ax] - p[ax]) / s[ax];
                float t1 = (hi[ax] - p[ax]) / s[ax];
                enter = max(enter, min(t0, t1));
                leave = min(leave, max(t0, t1));
            }
        }
        if (enter <= leave)
            return true;
    }
    return false;
}
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint width = asuint(params.right.w);
    uint height = asuint(params.up.w);
    if (tid.x >= width || tid.y >= height)
        return;
    uint flags = asuint(params.prev_fwd.w);
    uint max_age = max(asuint(params.prev_right.w), 1u);
    uint i = tid.y * width + tid.x;
    int3 px = int3(int(tid.x), int(tid.y), 0);
    float4 fresh = radiance[i];
    uint bad = fresh.w != 0.0 ? 1u : 0u;
    float depth = depth_tex.Load(px);

    if (depth <= 0.0)
    {
        // Sky: deterministic, passed through; no history.
        next_guides[i] = uint4(FACE_NONE, 0u, 0u, 0u);
        next_hist[i] = float4(fresh.xyz, 0.0);
        next_state[i] = state_bits(0u, R_SKY, bad);
        motion[i] = float2(0.0, 0.0);
        radiance[i] = float4(fresh.xyz, float(state_bits(0u, R_SKY, bad)));
        return;
    }

    // Current guide.
    float sx = 2.0 * (float(tid.x) + 0.5) / float(width) - 1.0;
    float sy = 1.0 - 2.0 * (float(tid.y) + 0.5) / float(height);
    float3 d = params.forward.xyz + sx * params.right.xyz + sy * params.up.xyz;
    float3 p = params.eye.xyz + d * (params.eye.w / depth);
    uint face = face_of(normal_tex.Load(px));
    uint axis = face / 2u;
    int plane = int(round(p[axis]));
    uint material = material_tex.Load(px);
    uint region = surface_tex.Load(px).x;
    uint rcount, rstride;
    regions.GetDimensions(rcount, rstride);
    int4 r = region < rcount ? regions[region] : int4(0, 0, 0, -1);
    uint4 cur = uint4(face | (material << 3), asuint(plane), pack_key(r.xyz), asuint(r.w));
    next_guides[i] = cur;

    // Reprojection into the previous frame.
    float3 v = p - params.prev_eye.xyz;
    float z = dot(v, params.prev_fwd.xyz);
    float2 prev_px = float2(-1.0e9, -1.0e9);
    bool on_screen = false;
    if (z > params.prev_eye.w)
    {
        float nx = dot(v, params.prev_right.xyz) / z;
        float ny = dot(v, params.prev_up.xyz) / z;
        prev_px = float2((nx + 1.0) * 0.5 * float(width), (1.0 - ny) * 0.5 * float(height));
        on_screen = nx >= -1.0 && nx <= 1.0 && ny >= -1.0 && ny <= 1.0;
    }
    float2 cur_px = float2(float(tid.x) + 0.5, float(tid.y) + 0.5);
    motion[i] = on_screen ? prev_px - cur_px : float2(0.0, 0.0);

    uint reason = R_ACCEPTED;
    float3 hist = float3(0.0, 0.0, 0.0);
    float m2 = 0.0;
    float age_w = 0.0;
    float wsum = 0.0;
    float4 lo_h = float4(1.0e30, 1.0e30, 1.0e30, 1.0e30);
    float4 hi_h = float4(-1.0e30, -1.0e30, -1.0e30, -1.0e30);
    bool sharp = false;
    float3 sharp_hist = float3(0.0, 0.0, 0.0);
    float sharp_m2 = 0.0;
    bool reset = (flags & (F_RESET | F_FAULT_RESET_ALWAYS)) != 0u;
    if (reset)
        reason = R_RESET;
    else if (!on_screen)
        reason = R_OFFSCREEN;
    else
    {
        // Bilinear taps around the previous pixel centre.
        float2 f = prev_px - 0.5;
        int2 base = int2(floor(f));
        float2 t = f - float2(base);
        // Within 1/1000 px of a pixel centre the tap is that pixel alone (ADR-0006): a still camera
        // reprojects to about 1e-5 px off the centre in f32, which would otherwise mix neighbours.
        t = select(t < SNAP, float2(0.0, 0.0), select(t > 1.0 - SNAP, float2(1.0, 1.0), t));
        uint nearest_reason = R_NO_PREV;
        float nearest_w = -1.0;
        for (uint k = 0u; k < 4u; k++)
        {
            int2 o = int2(int(k & 1u), int(k >> 1));
            int2 q = base + o;
            float w = (o.x == 1 ? t.x : 1.0 - t.x) * (o.y == 1 ? t.y : 1.0 - t.y);
            uint tr = R_OFFSCREEN;
            if (q.x >= 0 && q.y >= 0 && q.x < int(width) && q.y < int(height))
            {
                uint qi = uint(q.y) * width + uint(q.x);
                tr = check_tap(cur, prev_guides[qi], (flags & F_FAULT_NEVER_REJECT) != 0u);
                if (tr == R_ACCEPTED && w > 0.0)
                {
                    float4 ph = with_variance(prev_hist[qi]);
                    hist += w * ph.xyz;
                    m2 += w * ph.w;
                    lo_h = min(lo_h, ph);
                    hi_h = max(hi_h, ph);
                    age_w += w * float(prev_state[qi] & 0xFFFFu);
                    wsum += w;
                }
            }
            if (w > nearest_w)
            {
                nearest_w = w;
                nearest_reason = tr;
            }
        }
        if (wsum <= 0.0)
            reason = nearest_reason == R_ACCEPTED ? R_NO_PREV : nearest_reason;
        else if (relit(p))
            reason = R_RELIT;
#ifdef EMITTERS
        else if (relit_by_light(p, dot(hist / wsum, float3(0.2126, 0.7152, 0.0722))))
            reason = R_RELIT_LIGHT;
#endif
        // Catmull-Rom over the 4x4 taps, only when all of them with a weight are valid.
        if (reason == R_ACCEPTED && (flags & F_BILINEAR) == 0u)
        {
            float4 wx = catmull_rom(t.x);
            float4 wy = catmull_rom(t.y);
            bool ok = true;
            float4 acc = float4(0.0, 0.0, 0.0, 0.0);
            for (uint k = 0u; k < 16u && ok; k++)
            {
                uint kx = k & 3u;
                uint ky = k >> 2;
                float w = wx[kx] * wy[ky];
                if (w == 0.0)
                    continue;
                int2 q = base + int2(int(kx) - 1, int(ky) - 1);
                if (q.x < 0 || q.y < 0 || q.x >= int(width) || q.y >= int(height))
                {
                    ok = false;
                    continue;
                }
                uint qi = uint(q.y) * width + uint(q.x);
                if (check_tap(cur, prev_guides[qi], (flags & F_FAULT_NEVER_REJECT) != 0u) != R_ACCEPTED)
                {
                    ok = false;
                    continue;
                }
                acc += w * with_variance(prev_hist[qi]);
            }
            if (ok)
            {
                acc = clamp(acc, lo_h, hi_h);
                sharp = true;
                sharp_hist = acc.xyz;
                sharp_m2 = acc.w;
            }
        }
    }

    uint age;
    float3 c;
    float lum = dot(fresh.xyz, float3(0.2126, 0.7152, 0.0722));
    float moment = lum * lum;
    if (reason == R_ACCEPTED)
    {
        hist = sharp ? sharp_hist : hist / wsum;
        m2 = sharp ? sharp_m2 : m2 / wsum;
        float hl = dot(hist, float3(0.2126, 0.7152, 0.0722));
        m2 += hl * hl;
        uint prev_age = uint(age_w / wsum + 0.5);
        age = min(prev_age + 1u, max_age);
        c = (flags & F_FAULT_NO_FRESH) != 0u ? hist : hist + (fresh.xyz - hist) / float(age);
        moment = m2 + (moment - m2) / float(age);
    }
    else
    {
        age = 1u;
        c = fresh.xyz;
    }
    uint st = state_bits(age, reason, bad);
    next_hist[i] = float4(c, moment);
    next_state[i] = st;
    radiance[i] = float4(c, float(st));
}
