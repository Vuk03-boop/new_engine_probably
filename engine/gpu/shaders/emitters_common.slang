// Emitter sampling shared by `reference.slang` (4A) and, later, the real-time path (4B); ADR-0005
// Amendment 3. Host mirror: `light::emitters` (the alias draw, `uniform_index`, `SphericalRect`).
// Included after light_common.slang; it declares no resources.

static const float SOLID_ANGLE_MIN = 1.0e-2;

struct Emitter
{
    float4 p0_area;  // xyz corner (u0, v0), voxels; w area
    float4 eu_pdf;   // xyz edge along u; w realized selection probability
    float4 ev;       // xyz edge along v
    float4 radiance; // rgb emitted radiance
    uint4 meta;      // x alias threshold (0..2^24), y alias index, z face (ADR-0004)
};

uint next_u32(inout uint s)
{
    s = s * 747796405u + 2891336453u;
    uint word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

// The 32 x 32 -> 64-bit product as (high word, low word), from 16-bit halves (no Int64 capability).
uint2 mul_wide(uint a, uint b)
{
    uint a0 = a & 0xFFFFu;
    uint a1 = a >> 16u;
    uint b0 = b & 0xFFFFu;
    uint b1 = b >> 16u;
    uint p00 = a0 * b0;
    uint p01 = a0 * b1;
    uint p10 = a1 * b0;
    uint p11 = a1 * b1;
    uint mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    return uint2(p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u), (mid << 16u) | (p00 & 0xFFFFu));
}

// Uniform in 0..n (n >= 1), exactly: Lemire's multiply-shift with rejection.
uint uniform_index(uint n, inout uint s)
{
    uint2 m = mul_wide(next_u32(s), n);
    if (m.y < n)
    {
        uint t = (0u - n) % n;
        while (m.y < t)
            m = mul_wide(next_u32(s), n);
    }
    return m.x;
}

float3 face_normal(uint face)
{
    float3 n = float3(0.0, 0.0, 0.0);
    n[face / 2u] = (face & 1u) != 0u ? 1.0 : -1.0;
    return n;
}

// A rectangle seen from a point (Urena, Fajardo and King 2013), for sampling uniform in solid angle.
struct SphRect
{
    float3 o;
    float3 x;
    float3 y;
    float3 z;
    float z0;
    float x0;
    float y0;
    float x1;
    float y1;
    float b0;
    float b1;
    float k;
    float solid_angle;
};

float dihedral(float3 a, float3 b)
{
    return acos(clamp(-dot(a, b), -1.0, 1.0));
}

SphRect sph_rect(float3 p0, float3 eu, float3 ev, float3 o)
{
    SphRect r;
    r.o = o;
    float exl = length(eu);
    float eyl = length(ev);
    r.x = eu / exl;
    r.y = ev / eyl;
    r.z = cross(r.x, r.y);
    float3 d = p0 - o;
    r.z0 = dot(d, r.z);
    if (r.z0 > 0.0)
    {
        r.z = -r.z;
        r.z0 = -r.z0;
    }
    r.x0 = dot(d, r.x);
    r.y0 = dot(d, r.y);
    r.x1 = r.x0 + exl;
    r.y1 = r.y0 + eyl;
    float3 n0 = normalize(float3(0.0, r.z0, -r.y0));
    float3 n1 = normalize(float3(-r.z0, 0.0, r.x1));
    float3 n2 = normalize(float3(0.0, -r.z0, r.y1));
    float3 n3 = normalize(float3(r.z0, 0.0, -r.x0));
    float g0 = dihedral(n0, n1);
    float g1 = dihedral(n1, n2);
    float g2 = dihedral(n2, n3);
    float g3 = dihedral(n3, n0);
    r.b0 = n0.z;
    r.b1 = n2.z;
    r.k = 2.0 * PI - g2 - g3;
    r.solid_angle = g0 + g1 - r.k;
    return r;
}

float3 sph_rect_sample(SphRect r, float u, float v)
{
    float au = u * r.solid_angle + r.k;
    float fu = (cos(au) * r.b0 - r.b1) / sin(au);
    float cu = clamp((fu > 0.0 ? 1.0 : -1.0) / sqrt(fu * fu + r.b0 * r.b0), -1.0, 1.0);
    float xu = clamp(-(cu * r.z0) / sqrt(max(1.0 - cu * cu, 0.0)), r.x0, r.x1);
    float d = sqrt(xu * xu + r.z0 * r.z0);
    float h0 = r.y0 / sqrt(d * d + r.y0 * r.y0);
    float h1 = r.y1 / sqrt(d * d + r.y1 * r.y1);
    float hv = h0 + v * (h1 - h0);
    float yv = hv * hv < 1.0 - 1.0e-6 ? hv * d / sqrt(1.0 - hv * hv) : r.y1;
    return r.o + xu * r.x + yv * r.y + r.z0 * r.z;
}
