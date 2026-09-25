// Phase 2B (amended with ADR-0004 amendment 1): decodes the region mesh buffers on the GPU, so
// host/shader agreement on the packed layout is checked by execution, not only by reading the source.
//
// Quad record (uint2 = 64 bits, little-endian word order), region-local:
//   material [0,16)  face [16,19)  plane [19,26)  u0 [26,33)  v0 [33,40)  u1 [40,47)  v1 [47,54)
// Vertices: R16G16B16A16_SFLOAT (read here as uint2 and converted exactly), multiples of 1/2.
// Indices: 3 x u32 per triangle. Triangle quads: 1 x u32 per triangle.
//
// Thread i decodes quad i (if i < quad_count) and checks triangle i (if i < tri_count).

static const uint MATERIAL_SHIFT = 0;
static const uint FACE_SHIFT = 16;
static const uint PLANE_SHIFT = 19;
static const uint U0_SHIFT = 26;
static const uint V0_SHIFT = 33;
static const uint U1_SHIFT = 40;
static const uint V1_SHIFT = 47;

struct DecodedQuad
{
    uint material;
    uint face;
    uint plane;
    uint u0;
    uint v0;
    uint u1;
    uint v1;
    uint pad;
};

struct TriCheck
{
    // tri_quad[i] as read.
    uint quad;
    // Bits 0-2: vertex k lies on the quad's plane and rectangle, w = 1. Bit 3: the quad index is in
    // range. Bit 4: the winding faces the quad's outward normal (non-degenerate).
    uint ok;
};

struct Params
{
    uint quad_count;
    uint tri_count;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint2> quads;
[[vk::binding(1, 0)]] StructuredBuffer<uint2> vertices;
[[vk::binding(2, 0)]] StructuredBuffer<uint> indices;
[[vk::binding(3, 0)]] StructuredBuffer<uint> tri_quad;
[[vk::binding(4, 0)]] RWStructuredBuffer<DecodedQuad> decoded;
[[vk::binding(5, 0)]] RWStructuredBuffer<TriCheck> tri_checks;
[[vk::push_constant]] ConstantBuffer<Params> params;

uint field(uint2 q, uint shift, uint bits)
{
    // 64-bit extract from two 32-bit words.
    uint lo = shift < 32 ? (q.x >> shift) | (shift == 0 ? 0u : (q.y << (32 - shift))) : q.y >> (shift - 32);
    return lo & ((1u << bits) - 1u);
}

// IEEE half to float with 32-bit integer operations only, so the module declares no 16-bit
// capability (Int16/Float16 would need the shaderInt16/shaderFloat16 device features).
float half_to_float(uint h)
{
    uint e = (h >> 10) & 0x1F;
    uint m = h & 0x3FF;
    float v;
    if (e == 0)
        v = float(m) * exp2(-24.0); // zero and subnormals, exact
    else if (e == 31)
        v = asfloat(0x7F800000u | (m << 13)); // inf / NaN
    else
        v = asfloat(((e + 112) << 23) | (m << 13));
    return (h & 0x8000) != 0 ? -v : v;
}

DecodedQuad decode(uint2 q)
{
    DecodedQuad d;
    d.material = field(q, MATERIAL_SHIFT, 16);
    d.face = field(q, FACE_SHIFT, 3);
    d.plane = field(q, PLANE_SHIFT, 7);
    d.u0 = field(q, U0_SHIFT, 7);
    d.v0 = field(q, V0_SHIFT, 7);
    d.u1 = field(q, U1_SHIFT, 7);
    d.v1 = field(q, V1_SHIFT, 7);
    d.pad = 0;
    return d;
}

float4 vertex(uint i)
{
    uint2 v = vertices[i];
    return float4(half_to_float(v.x & 0xFFFF), half_to_float(v.x >> 16), half_to_float(v.y & 0xFFFF), half_to_float(v.y >> 16));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i < params.quad_count)
        decoded[i] = decode(quads[i]);
    if (i >= params.tri_count)
        return;
    TriCheck c;
    c.quad = tri_quad[i];
    c.ok = 0;
    if (c.quad < params.quad_count)
    {
        c.ok |= 8;
        DecodedQuad d = decode(quads[c.quad]);
        uint a = d.face / 2;
        uint ua = (a + 1) % 3;
        uint va = (a + 2) % 3;
        float3 p[3];
        for (uint k = 0; k < 3; k++)
        {
            float4 v = vertex(indices[3 * i + k]);
            p[k] = v.xyz;
            bool on = v[a] == float(d.plane) && v[ua] >= float(d.u0) && v[ua] <= float(d.u1) && v[va] >= float(d.v0) && v[va] <= float(d.v1) && v.w == 1.0;
            if (on)
                c.ok |= 1u << k;
        }
        float3 n = cross(p[1] - p[0], p[2] - p[0]);
        float along = (d.face & 1) != 0 ? n[a] : -n[a];
        if (along > 0.0)
            c.ok |= 16;
    }
    tri_checks[i] = c;
}
