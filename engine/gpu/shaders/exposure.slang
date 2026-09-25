// Phase 4B: the sums behind the viewer's automatic exposure (Phase 4 decision 3). `groups` workgroups
// of 256 threads read every `stride`-th pixel in x and y (from stride / 2) of the shown radiance and
// adds, on surface pixels with EMISSION, the material's emitted radiance (what the light view shows,
// before exposure). Per frame and workgroup g, into `sums[slot * groups + g]` (the host adds the
// groups' partial sums): x the sum of the displayed luminance Y, y the number
// of pixels read, z the sum of ln Y over the pixels with Y >= floor and Y > 0 (the 4A metric
// exposure's rule, with the host's floor from the previous frame's mean), w their number. The host
// reads it once the frame has completed (`gpu::exposure`).

struct Params
{
    uint width;
    uint height;
    uint stride;
    uint slot;
    float floor;
    uint flags;
    uint material_count;
    uint groups;
};

struct MaterialRow
{
    float4 base;
    float4 emissive;
};

[[vk::binding(0, 0)]] [[vk::image_format("unknown")]] Texture2D<float> depth_tex;
[[vk::binding(1, 0)]] [[vk::image_format("unknown")]] Texture2D<uint> material_tex;
[[vk::binding(2, 0)]] StructuredBuffer<float4> radiance;
[[vk::binding(3, 0)]] StructuredBuffer<MaterialRow> materials;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> sums;
[[vk::push_constant]] ConstantBuffer<Params> params;

static const uint F_EMISSION = 1u;
static const uint THREADS = 256u;

groupshared float4 part[THREADS];

float luminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint t = tid.x;
    uint groups = max(params.groups, 1u);
    uint s = max(params.stride, 1u);
    uint nx = (params.width + s / 2u) / s;
    uint ny = (params.height + s / 2u) / s;
    uint n = nx * ny;
    float4 acc = float4(0.0, 0.0, 0.0, 0.0);
    for (uint j = gid.x * THREADS + t; j < n; j += groups * THREADS)
    {
        uint x = (j % nx) * s + s / 2u;
        uint y = (j / nx) * s + s / 2u;
        int3 px = int3(int(x), int(y), 0);
        float yl = luminance(radiance[y * params.width + x].xyz);
        if ((params.flags & F_EMISSION) != 0u && depth_tex.Load(px) > 0.0)
        {
            uint m = material_tex.Load(px);
            if (m < params.material_count)
                yl += luminance(materials[m].emissive.xyz);
        }
        acc.x += yl;
        acc.y += 1.0;
        if (yl >= params.floor && yl > 0.0)
        {
            acc.z += log(yl);
            acc.w += 1.0;
        }
    }
    part[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint h = THREADS / 2u; h > 0u; h /= 2u)
    {
        if (t < h)
            part[t] += part[t + h];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0u)
        sums[params.slot * groups + gid.x] = part[0];
}
