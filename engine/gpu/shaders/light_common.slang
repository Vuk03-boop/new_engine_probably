// Light functions shared by `reference.slang` (3A) and `shade.slang` (3B), ADR-0005. Host mirrors:
// `light::sample` (RNG, samplers) and `light::atmosphere` (the atmosphere and its Monte Carlo sky).
// Included with #include; it declares no resources.

static const float PI = 3.14159265358979;
static const float RAY_OFFSET = 1.0 / 256.0;
static const uint MAX_EVENTS = 32u;
static const uint RR_FROM = 3u;

// ADR-0005 atmosphere (host mirror: light::Atmosphere::default()).
static const float R_GROUND = 6360000.0;
static const float TOP_SQ_MINUS_GROUND_SQ = 100000.0 * 12820000.0; // (Rt - R)(Rt + R)
static const float3 RAYLEIGH = float3(5.802e-6, 13.558e-6, 33.1e-6);
static const float H_RAYLEIGH = 8000.0;
static const float MIE_SCATTERING = 3.996e-6;
static const float MIE_EXTINCTION = 4.440e-6;
static const float H_MIE = 1200.0;
static const float MIE_G = 0.8;
static const float3 OZONE = float3(0.650e-6, 1.881e-6, 0.085e-6);
static const float OZONE_CENTER = 25000.0;
static const float OZONE_HALF_WIDTH = 15000.0;
static const float GROUND_ALBEDO = 0.3;
static const float OBSERVER_ALTITUDE = 2.0;

// ---- random numbers (light::sample) ----

uint pcg_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint s)
{
    s = s * 747796405u + 2891336453u;
    uint word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word >> 8u) * (1.0 / 16777216.0);
}

float3 local_dir(float3 n, float x, float y, float z)
{
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    float3 t = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    float3 bt = float3(b, s + n.y * n.y * a, -n.y);
    return t * x + bt * y + n * z;
}

float3 cosine_hemisphere(float3 n, float u1, float u2)
{
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    return local_dir(n, r * cos(phi), r * sin(phi), sqrt(max(1.0 - u1, 0.0)));
}

float3 uniform_cone(float3 axis, float cos_max, float u1, float u2)
{
    float c = 1.0 - u1 * (1.0 - cos_max);
    float s = sqrt(max(1.0 - c * c, 0.0));
    float phi = 2.0 * PI * u2;
    return local_dir(axis, s * cos(phi), s * sin(phi), c);
}

float3 uniform_sphere(float u1, float u2)
{
    float z = 1.0 - 2.0 * u1;
    float s = sqrt(max(1.0 - z * z, 0.0));
    float phi = 2.0 * PI * u2;
    return float3(s * cos(phi), s * sin(phi), z);
}

float hg_phase(float c, float g)
{
    float d = 1.0 + g * g - 2.0 * g * c;
    return (1.0 - g * g) / (4.0 * PI * d * sqrt(d));
}

float3 sample_hg(float3 dir, float g, float u1, float u2)
{
    float s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
    float c = clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
    float sn = sqrt(max(1.0 - c * c, 0.0));
    float phi = 2.0 * PI * u2;
    return local_dir(dir, sn * cos(phi), sn * sin(phi), c);
}

float rayleigh_phase(float c)
{
    return 3.0 / (16.0 * PI) * (1.0 + c * c);
}

// ---- atmosphere (light::atmosphere) ----

float c_ground(float3 q)
{
    return dot(q, q) + 2.0 * R_GROUND * q.y;
}

float altitude(float3 q)
{
    float y = R_GROUND + q.y;
    return c_ground(q) / (sqrt(q.x * q.x + q.z * q.z + y * y) + R_GROUND);
}

float bdot(float3 q, float3 d)
{
    return dot(q, d) + R_GROUND * d.y;
}

bool ground_hit(float3 q, float3 d, out float t)
{
    float b = bdot(q, d);
    float c = c_ground(q);
    float disc = b * b - c;
    t = 0.0;
    if (b < 0.0 && disc >= 0.0)
    {
        t = max(c / (-b + sqrt(disc)), 0.0);
        return true;
    }
    return false;
}

float top_exit(float3 q, float3 d)
{
    float b = bdot(q, d);
    float c = c_ground(q) - TOP_SQ_MINUS_GROUND_SQ;
    return -b + sqrt(max(b * b - c, 0.0));
}

struct Medium
{
    float3 rayleigh;
    float mie;
    float3 extinction;
};

Medium medium(float h)
{
    Medium m;
    m.rayleigh = RAYLEIGH * exp(-h / H_RAYLEIGH);
    float md = exp(-h / H_MIE);
    m.mie = MIE_SCATTERING * md;
    float o = max(1.0 - abs(h - OZONE_CENTER) / OZONE_HALF_WIDTH, 0.0);
    m.extinction = m.rayleigh + MIE_EXTINCTION * md + OZONE * o;
    return m;
}

float step_bound(uint i, uint n, float t_end, float t_c)
{
    if (t_c > 0.0 && t_c < t_end)
    {
        uint hn = n / 2u;
        if (i <= hn)
        {
            float u = float(i) / float(hn);
            return t_c * (1.0 - (1.0 - u) * (1.0 - u));
        }
        float u = float(i - hn) / float(hn);
        return t_c + (t_end - t_c) * u * u;
    }
    float u = float(i) / float(n);
    if (t_c <= 0.0)
        return t_end * u * u;
    return t_end * (1.0 - (1.0 - u) * (1.0 - u));
}

float3 transmittance(float3 q, float3 d, uint steps)
{
    float tg;
    if (ground_hit(q, d, tg))
        return float3(0.0, 0.0, 0.0);
    float t_end = top_exit(q, d);
    float t_c = -bdot(q, d);
    float3 tau = float3(0.0, 0.0, 0.0);
    float t0 = 0.0;
    for (uint i = 1u; i <= steps; i++)
    {
        float t1 = step_bound(i, steps, t_end, t_c);
        Medium m = medium(altitude(q + d * (0.5 * (t0 + t1))));
        tau += m.extinction * (t1 - t0);
        t0 = t1;
    }
    return exp(-tau);
}

float3 in_scatter(Medium m, float c)
{
    return m.rayleigh * rayleigh_phase(c) + m.mie * hg_phase(c, MIE_G);
}

float mean3(float3 v)
{
    return (v.x + v.y + v.z) / 3.0;
}

// One sample of the sky radiance from the observer along unit d (sun disk excluded).
float3 sky_sample(float3 d, float3 sun, inout uint rng, uint steps, uint sun_steps)
{
    float3 q = float3(0.0, OBSERVER_ALTITUDE, 0.0);
    float3 beta = float3(1.0, 1.0, 1.0);
    float3 l = float3(0.0, 0.0, 0.0);
    for (uint event = 0u; event < MAX_EVENTS; event++)
    {
        float t_end;
        bool ground = ground_hit(q, d, t_end);
        if (!ground)
            t_end = top_exit(q, d);
        float t_c = -bdot(q, d);
        uint c = min(uint(rnd(rng) * 3.0), 2u);
        float target = -log(1.0 - rnd(rng));
        float3 tau = float3(0.0, 0.0, 0.0);
        float t0 = 0.0;
        bool scattered = false;
        float t_hit = 0.0;
        Medium mh;
        for (uint i = 1u; i <= steps; i++)
        {
            float t1 = step_bound(i, steps, t_end, t_c);
            Medium m = medium(altitude(q + d * (0.5 * (t0 + t1))));
            float next = tau[c] + m.extinction[c] * (t1 - t0);
            if (next >= target)
            {
                t_hit = t0 + (target - tau[c]) / m.extinction[c];
                tau += m.extinction * (t_hit - t0);
                mh = m;
                scattered = true;
                break;
            }
            tau += m.extinction * (t1 - t0);
            t0 = t1;
        }
        float3 tr = exp(-tau);
        if (scattered)
        {
            beta = beta * tr / mean3(mh.extinction * tr);
            q = q + d * t_hit;
            float3 ts = transmittance(q, sun, sun_steps);
            l += beta * in_scatter(mh, dot(d, sun)) * ts;
            float wr = mean3(mh.rayleigh) / (mean3(mh.rayleigh) + mh.mie);
            float3 nd;
            if (rnd(rng) < wr)
            {
                float u1 = rnd(rng);
                nd = uniform_sphere(u1, rnd(rng));
            }
            else
            {
                float u1 = rnd(rng);
                nd = sample_hg(d, MIE_G, u1, rnd(rng));
            }
            float cs = dot(d, nd);
            float pdf_dir = wr / (4.0 * PI) + (1.0 - wr) * hg_phase(cs, MIE_G);
            beta = beta * in_scatter(mh, cs) / pdf_dir;
            d = nd;
        }
        else
        {
            beta = beta * tr / mean3(tr);
            if (!ground)
                break;
            q = q + d * t_end;
            float3 n = normalize(float3(q.x, q.y + R_GROUND, q.z));
            float cos_s = dot(n, sun);
            if (cos_s > 0.0)
                l += beta * transmittance(q, sun, sun_steps) * (GROUND_ALBEDO / PI * cos_s);
            float u1 = rnd(rng);
            d = cosine_hemisphere(n, u1, rnd(rng));
            beta *= GROUND_ALBEDO;
        }
        if (event + 1u >= RR_FROM)
        {
            float p = min(max(beta.x, max(beta.y, beta.z)), 1.0);
            if (p <= 0.0 || rnd(rng) >= p)
                break;
            beta /= p;
        }
    }
    return l;
}
