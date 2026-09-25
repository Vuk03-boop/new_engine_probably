// Phase 2C-1: rasterizes ADR-0004 region meshes into the ADR-0003 primary-visibility G-buffer.
//
// - Vertices: R16G16B16A16_SFLOAT, region-local voxel units (exact multiples of 1/2).
// - Camera-relative: `region_offset` is the region origin minus the integer camera origin (exact
//   in f32); the fractional camera position is folded into the view-projection rows on the host.
// - Projection: reverse-Z infinite (clip z = near, clip w = view distance along forward), so
//   depth = near / distance. Culling and winding are pipeline state (host side).
// - Targets: normal (octahedral, R16G16_SNORM), material (R16_UINT), surface id (R32G32_UINT:
//   region index, PrimitiveID). Primitive p belongs to quad tri_quad[p] (ADR-0004 amendment 1).
//   Quad record layout: see mesh_decode.slang / gpu::layout.

struct Params
{
    // View-projection rows, applied to camera-relative positions.
    float4 row0;
    float4 row1;
    float4 row2;
    float4 row3;
    // xyz: region origin minus the integer camera origin, voxels. w unused.
    float4 region_offset;
    uint region_index;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint2> quads;
[[vk::binding(1, 0)]] StructuredBuffer<uint> tri_quad;
[[vk::push_constant]] ConstantBuffer<Params> params;

struct VsOut
{
    float4 pos : SV_Position;
};

[shader("vertex")]
VsOut vs_main(float4 position : POSITION)
{
    float4 p = float4(position.xyz + params.region_offset.xyz, 1.0);
    VsOut o;
    o.pos = float4(dot(params.row0, p), dot(params.row1, p), dot(params.row2, p), dot(params.row3, p));
    return o;
}

// Octahedral encoding; the host (gpu::equivalence::oct_encode) uses the same expression.
float2 oct_encode(float3 n)
{
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 p = n.xy;
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * float2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    return p;
}

struct FsOut
{
    float2 normal : SV_Target0;
    uint material : SV_Target1;
    uint2 surface : SV_Target2;
};

[shader("fragment")]
FsOut fs_main(uint prim : SV_PrimitiveID)
{
    uint2 q = quads[tri_quad[prim]];
    uint face = (q.x >> 16) & 7;
    float3 n = float3(0.0, 0.0, 0.0);
    n[face / 2] = (face & 1) != 0 ? 1.0 : -1.0;
    FsOut o;
    o.normal = oct_encode(n);
    o.material = q.x & 0xFFFF;
    o.surface = uint2(params.region_index, prim);
    return o;
}
