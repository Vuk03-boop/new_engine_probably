// Phase 2C-2: debug views of the ADR-0003 G-buffer, drawn as one fullscreen triangle into the
// swapchain image. Reads the targets with Load (no sampler, no filtering: every value is exact).
// M1 adds the lit view: simple sun and ambient shading from the same G-buffer values.
//
// Views (`params.origin.w`), in gpu::debug_view::View order:
//   0 material colour (MaterialParams base colour, shaded by face axis only so faces are legible)
//   1 normal (decoded from the octahedral target)
//   2 region hash (region key)
//   3 brick hash (brick of the voxel behind the surface, reconstructed from depth)
//   4 surface-id hash (region index, PrimitiveID)
//   5 linear depth (view distance, log scale, 1 to 4096 voxels)
//   6 snapshot version (the 1C snapshot each region was uploaded from)
//   7 lit: base colour x (sun x max(N.L, 0) + hemisphere ambient) + emissive, exposure, then the
//     Narkowicz ACES fit. No shadows, no occlusion. Output is linear (the sRGB image encodes).
//   8 light (3B): the real-time lighting of gpu::shade (linear HDR radiance per pixel, row-major in
//     `radiance`), times params.light.w (exposure), then the same ACES fit. Covers every pixel,
//     surfaces and sky alike.
//   9 history age, 10 history reason (3D; 3E adds relit, 4B relit by a light): the temporal state gpu::temporal writes into radiance.w
//     (the number age | reason << 16 | bad << 24, stored as a float value).
//   11 motion (3D): the motion buffer, pixels to the previous frame.
// Background (nothing drawn) is a fixed sky colour in the debug views, a gradient in the lit view.
// 2E: `params.light.z` selects the source. 0 reads the raster G-buffer images; 1 reads the ray-query
// outputs (gpu::ray: per pixel a Hit {depth bits, normal SNORM bits, material, pad} and a surface id),
// which hold the same values, so every view shows either path unchanged.

struct Params
{
    // Camera basis, pre-scaled: right * tan_half_x, up * tan_half_y, forward.
    float4 right;
    float4 up;
    float4 forward;
    // xyz: eye minus the integer camera origin; w: near plane.
    float4 eye;
    // xyz: integer camera origin; w: view.
    int4 origin;
    // x, y: target size; z: material count; w: region count.
    uint4 info;
    // xyz: unit vector toward the sun; w: sun intensity.
    float4 sun;
    // x: ambient intensity; y: exposure; z: source (0 raster, 1 ray query); w: light-view exposure.
    float4 light;
};

struct MaterialRow
{
    // Linear base colour, w unused.
    float4 base;
    // Emissive, same relative units, w unused.
    float4 emissive;
};

// Lit-view constants (mirrored by the CPU check in gpu/tests/raster.rs).
static const float3 SUN_COLOUR = float3(1.0, 0.95, 0.88);
static const float3 SKY_AMBIENT = float3(0.55, 0.7, 1.0);
static const float3 GROUND_AMBIENT = float3(0.35, 0.3, 0.25);
static const float3 SKY_ZENITH = float3(0.25, 0.45, 0.85);
static const float3 SKY_HORIZON = float3(0.7, 0.8, 0.92);

// Sampled images must carry the SPIR-V format Unknown: Slang otherwise derives one from the texel
// type (e.g. R32ui for `uint`), which differs from the views (R16_UINT, R32G32_UINT) and makes loads
// undefined; the two-component ones would also need StorageImageExtendedFormats.
[[vk::binding(0, 0)]] [[vk::image_format("unknown")]] Texture2D<float> depth_tex;
[[vk::binding(1, 0)]] [[vk::image_format("unknown")]] Texture2D<float2> normal_tex;
[[vk::binding(2, 0)]] [[vk::image_format("unknown")]] Texture2D<uint> material_tex;
[[vk::binding(3, 0)]] [[vk::image_format("unknown")]] Texture2D<uint2> surface_tex;
// Per MaterialId.
[[vk::binding(4, 0)]] StructuredBuffer<MaterialRow> materials;
// Per region index: xyz region key, w snapshot id.
[[vk::binding(5, 0)]] StructuredBuffer<int4> regions;
// Ray-query outputs, row-major (read only when the source is 1).
[[vk::binding(6, 0)]] StructuredBuffer<uint4> ray_hits;
[[vk::binding(7, 0)]] StructuredBuffer<uint2> ray_surfaces;
// Real-time lighting (view 8), row-major, the targets' size.
[[vk::binding(8, 0)]] StructuredBuffer<float4> radiance;
// Motion to the previous frame (view 11), row-major.
[[vk::binding(9, 0)]] StructuredBuffer<float2> motion;
[[vk::push_constant]] ConstantBuffer<Params> params;

struct VsOut
{
    float4 pos : SV_Position;
};

[shader("vertex")]
VsOut vs_main(uint id : SV_VulkanVertexID)
{
    // One triangle covering the screen: (-1,-1), (3,-1), (-1,3). SV_VulkanVertexID is VertexIndex
    // itself (SV_VertexID would subtract BaseVertex and need the DrawParameters capability).
    float2 p = float2((id << 1) & 2, id & 2) * 2.0 - 1.0;
    VsOut o;
    o.pos = float4(p, 0.0, 1.0);
    return o;
}

// Same decoding as gpu::equivalence::oct_encode's inverse.
float3 oct_decode(float2 e)
{
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

// 16-bit SNORM bits to float, as the R16G16_SNORM target reads.
float snorm16_to_float(uint bits)
{
    return max(float(int(bits << 16) >> 16) / 32767.0, -1.0);
}

uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

float3 hash_colour(uint h)
{
    h = hash(h);
    return float3(float(h & 255), float((h >> 8) & 255), float((h >> 16) & 255)) / 255.0 * 0.8 + 0.2;
}

float3 aces(float3 x)
{
    return saturate(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14));
}

// Pixel centre ray direction, as gpu::raster::Camera::dir: forward component 1.
float3 pixel_dir(float4 frag)
{
    float2 ndc = float2(2.0 * frag.x / float(params.info.x) - 1.0, 1.0 - 2.0 * frag.y / float(params.info.y));
    return params.forward.xyz + ndc.x * params.right.xyz + ndc.y * params.up.xyz;
}

uint hash3(int3 k)
{
    return hash(uint(k.x) * 73856093u ^ hash(uint(k.y) * 19349663u ^ hash(uint(k.z) * 83492791u)));
}

[shader("fragment")]
float4 fs_main(float4 frag : SV_Position) : SV_Target0
{
    int2 px = int2(frag.xy);
    uint view = uint(params.origin.w);
    if (view == 8)
    {
        // Bounds-checked: without a bound radiance buffer the material table stands in (magenta).
        uint ri = uint(px.y) * params.info.x + uint(px.x);
        uint rcount, rstride;
        radiance.GetDimensions(rcount, rstride);
        if (ri >= rcount)
            return float4(1.0, 0.0, 1.0, 1.0);
        return float4(aces(radiance[ri].xyz * params.light.w), 1.0);
    }
    if (view == 9 || view == 10)
    {
        uint ri = uint(px.y) * params.info.x + uint(px.x);
        uint rcount, rstride;
        radiance.GetDimensions(rcount, rstride);
        if (ri >= rcount)
            return float4(1.0, 0.0, 1.0, 1.0);
        uint st = uint(radiance[ri].w);
        uint age = st & 0xFFFFu;
        uint reason = (st >> 16) & 0xFFu;
        if (view == 9)
        {
            if (reason == 1u)
                return float4(0.1, 0.1, 0.1, 1.0);
            float a = saturate(float(age) / 64.0);
            return float4(a * a, a, 0.3 + 0.7 * a, 1.0);
        }
        // accepted, sky, off-screen, no previous, material, normal, disoccluded, edited, reset, relit,
        // relit by a light (4B)
        float3 REASON[11] = { float3(0.2, 0.6, 0.2), float3(0.1, 0.1, 0.1), float3(0.9, 0.9, 0.2), float3(0.5, 0.2, 0.7),
            float3(1.0, 0.5, 0.0), float3(0.0, 0.7, 1.0), float3(1.0, 0.1, 0.1), float3(1.0, 0.2, 0.9), float3(1.0, 1.0, 1.0),
            float3(0.2, 0.3, 1.0), float3(1.0, 0.85, 0.45) };
        return float4(reason < 11u ? REASON[reason] : float3(1.0, 0.0, 1.0), 1.0);
    }
    if (view == 11)
    {
        uint ri = uint(px.y) * params.info.x + uint(px.x);
        uint mcount, mstride;
        motion.GetDimensions(mcount, mstride);
        if (ri >= mcount)
            return float4(1.0, 0.0, 1.0, 1.0);
        float2 m = motion[ri];
        float len = saturate(length(m) / 16.0);
        float2 dir = length(m) > 0.0 ? normalize(m) : float2(0.0, 0.0);
        return float4(len * (0.5 + 0.5 * dir.x), len * (0.5 + 0.5 * dir.y), len * 0.5, 1.0);
    }
    float d;
    uint2 surface;
    float2 oct;
    uint material;
    if (params.light.z > 0.5)
    {
        uint i = uint(px.y) * params.info.x + uint(px.x);
        uint4 h = ray_hits[i];
        d = asfloat(h.x);
        oct = float2(snorm16_to_float(h.y & 0xFFFFu), snorm16_to_float(h.y >> 16));
        material = h.z;
        surface = ray_surfaces[i];
    }
    else
    {
        d = depth_tex.Load(int3(px, 0));
        oct = normal_tex.Load(int3(px, 0));
        material = material_tex.Load(int3(px, 0));
        surface = surface_tex.Load(int3(px, 0));
    }
    if (d <= 0.0 || surface.x >= params.info.w)
    {
        if (view == 7)
        {
            float3 dir = normalize(pixel_dir(frag));
            return float4(aces(lerp(SKY_HORIZON, SKY_ZENITH, saturate(dir.y)) * params.light.y), 1.0);
        }
        return float4(0.45, 0.62, 0.85, 1.0);
    }
    float3 n = oct_decode(oct);
    float3 axis_shade = float3(0.75, 1.0, 0.6); // x, y, z faces
    float shade = abs(n.x) * axis_shade.x + abs(n.y) * axis_shade.y + abs(n.z) * axis_shade.z;
    float3 c = float3(1.0, 0.0, 1.0);
    if (view == 0)
    {
        uint m = material;
        c = m < params.info.z ? materials[m].base.xyz * shade : float3(1.0, 0.0, 1.0);
    }
    else if (view == 1)
    {
        c = n * 0.5 + 0.5;
    }
    else if (view == 2)
    {
        c = hash_colour(hash3(regions[surface.x].xyz)) * shade;
    }
    else if (view == 3)
    {
        // Forward component 1, so t = distance.
        float3 dir = pixel_dir(frag);
        float t = params.eye.w / d;
        // Half a voxel back along the normal: inside the solid voxel that owns the face.
        float3 p = params.eye.xyz + t * dir - 0.5 * n;
        int3 voxel = int3(floor(p)) + params.origin.xyz;
        int3 brick = int3(floor(float3(voxel) / 8.0));
        c = hash_colour(hash3(brick)) * shade;
    }
    else if (view == 4)
    {
        c = hash_colour(hash(surface.x * 0x9E3779B9u) ^ surface.y) * shade;
    }
    else if (view == 5)
    {
        float t = params.eye.w / d;
        float g = 1.0 - saturate(log2(max(t, 1.0)) / 12.0);
        c = float3(g, g, g);
    }
    else if (view == 6)
    {
        c = hash_colour(uint(regions[surface.x].w) + 1u) * shade;
    }
    else if (view == 7)
    {
        uint m = material;
        if (m >= params.info.z)
            return float4(1.0, 0.0, 1.0, 1.0);
        MaterialRow row = materials[m];
        float3 ambient = lerp(GROUND_AMBIENT, SKY_AMBIENT, n.y * 0.5 + 0.5) * params.light.x;
        float3 direct = SUN_COLOUR * (params.sun.w * max(dot(n, params.sun.xyz), 0.0));
        c = aces((row.base.xyz * (direct + ambient) + row.emissive.xyz) * params.light.y);
    }
    return float4(c, 1.0);
}
