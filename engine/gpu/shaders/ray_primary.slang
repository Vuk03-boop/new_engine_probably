// Phase 2D: primary visibility by ray query, into the same values as the ADR-0003 G-buffer, so the
// 2C-1 per-pixel check (ADR-0003 Amendment 1) applies to both paths unchanged.
//
// - One ray per pixel, through the pixel centre (x + 0.5, y + 0.5), with the camera basis of
//   `gpu::raster::Camera`: dir = forward + sx * right + sy * up, so its forward component is 1 and
//   the hit distance t is the view distance. Depth is written as near / t (reverse-Z), like raster.
// - TMin is `near`: raster clips surfaces closer than the near plane, and so does this pass.
// - Opaque, back faces culled (raster culls them too). Vulkan's default ray-triangle facing treats
//   ADR-0004's winding as front-facing, so instances carry no facing flag.
// - Surface id: (InstanceCustomIndex, PrimitiveIndex) = (region index, triangle index), the pair
//   raster writes (ADR-0003). Material and normal come from the region's quad table through
//   `tri_quad`, read by buffer device address from the region table. Every read is bounds-checked:
//   an out-of-range region, primitive or quad index writes material BAD_MATERIAL and keeps the
//   surface id, so the host check reports it (the first 2D run lost the device on a planted wrong id).
// - Misses write the raster clear values: depth 0, normal 0, material 0xFFFF, surface (~0, ~0).

struct Params
{
    // xyz: eye, world voxels (the TLAS is in world coordinates). w unused.
    float4 eye;
    float4 forward;
    // right * tan_half_x and up * tan_half_y.
    float4 right;
    float4 up;
    uint width;
    uint height;
    float near;
    uint pad;
};

// One per region, in region key order (the instance custom index).
struct RegionRef
{
    // Device addresses of the region's quad table (uint2 per quad) and triangle-quad map (uint per triangle).
    uint2* quads;
    uint* tri_quad;
    // Bounds of both tables: an id that does not fit them is written as a bad id, never read through.
    uint quad_count;
    uint tri_count;
};

struct Hit
{
    uint depth;    // f32 bits
    uint normal;   // R16G16_SNORM bits, x in the low half
    uint material;
    uint pad;
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] StructuredBuffer<RegionRef> regions;
[[vk::binding(2, 0)]] RWStructuredBuffer<Hit> hits;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint2> surfaces;
[[vk::push_constant]] ConstantBuffer<Params> params;

static const uint BAD_MATERIAL = 0xFFFEu;

// Octahedral encoding, the same expression as raster.slang and gpu::equivalence::oct_encode.
float2 oct_encode(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 p = n.xy;
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * float2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    return p;
}

// Float to 16-bit SNORM as the colour attachment stores it (round to nearest).
uint snorm16(float v)
{
    return uint(int(round(clamp(v, -1.0, 1.0) * 32767.0))) & 0xFFFFu;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= params.width || tid.y >= params.height)
        return;
    uint i = tid.y * params.width + tid.x;
    float sx = (2.0 * (float(tid.x) + 0.5) / float(params.width) - 1.0);
    float sy = (1.0 - 2.0 * (float(tid.y) + 0.5) / float(params.height));
    RayDesc ray;
    ray.Origin = params.eye.xyz;
    ray.Direction = params.forward.xyz + sx * params.right.xyz + sy * params.up.xyz;
    ray.TMin = params.near;
    ray.TMax = 1.0e30;

    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    Hit h;
    h.depth = 0;
    h.normal = 0;
    h.material = 0xFFFFu;
    h.pad = 0;
    uint2 surface = uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        uint region = q.CommittedInstanceID();
        uint prim = q.CommittedPrimitiveIndex();
        h.depth = asuint(params.near / q.CommittedRayT());
        h.material = BAD_MATERIAL;
        surface = uint2(region, prim);
        uint count, stride;
        regions.GetDimensions(count, stride);
        if (region < count)
        {
            RegionRef r = regions[region];
            uint quad = prim < r.tri_count ? r.tri_quad[prim] : 0xFFFFFFFFu;
            if (quad < r.quad_count)
            {
                uint2 qbits = r.quads[quad];
                uint face = (qbits.x >> 16) & 7;
                float3 n = float3(0.0, 0.0, 0.0);
                n[face / 2] = (face & 1) != 0 ? 1.0 : -1.0;
                float2 e = oct_encode(n);
                h.normal = snorm16(e.x) | (snorm16(e.y) << 16);
                h.material = qbits.x & 0xFFFFu;
            }
        }
    }
    hits[i] = h;
    surfaces[i] = surface;
}
