// Textured probe (rt_probe --textures): base-colour textures, alpha-tested geometry via any-hit,
// sRGB output. Same camera, sun and lighting model as probe.slang; that file stays the frozen control.
//
// BLAS geometry 0 = opaque triangles, geometry 1 = alpha-tested triangles (no OPAQUE flag).
// Triangle arrays are ordered [opaque..., alpha...]; cam.sun.w carries the opaque count as uint bits.
// Hit groups: 0 = primary (closest hit + alpha any-hit), 1 = shadow (alpha any-hit only).

struct Payload { float3 color; };
struct ShadowPayload { uint visible; };

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> img;
[[vk::binding(2, 0)]] StructuredBuffer<float4> normals;   // per triangle: geometric normal
[[vk::binding(3, 0)]] StructuredBuffer<float2> uvs;       // per triangle corner
[[vk::binding(4, 0)]] StructuredBuffer<uint> triTexture;  // per triangle: base-colour texture or ~0
[[vk::binding(5, 0)]] SamplerState samp;
[[vk::binding(6, 0)]] Texture2D textures[];

struct Camera { float4 origin; float4 forward; float4 right; float4 up; float4 sun; };
[[vk::push_constant]] ConstantBuffer<Camera> cam;

static const uint kNoTexture = 0xFFFFFFFFu;

uint triangleId() { return PrimitiveIndex() + (GeometryIndex() == 1 ? asuint(cam.sun.w) : 0u); }

float2 hitUv(uint tri, float2 bary) {
  return uvs[tri * 3] * (1.0 - bary.x - bary.y) + uvs[tri * 3 + 1] * bary.x + uvs[tri * 3 + 2] * bary.y;
}

// Mip 0 only: ray tracing has no screen derivatives and this probe has no ray cones (aliasing accepted).
float4 baseColor(uint tri, float2 bary) {
  uint t = triTexture[tri];
  if (t == kNoTexture) return float4(0.7, 0.7, 0.7, 1.0);
  return textures[NonUniformResourceIndex(t)].SampleLevel(samp, hitUv(tri, bary), 0.0);
}

float3 toSrgb(float3 c) { return pow(saturate(c), 1.0 / 2.2); }

[shader("raygeneration")]
void rgen() {
  uint2 id = DispatchRaysIndex().xy;
  uint2 size = DispatchRaysDimensions().xy;
  float2 uv = (float2(id) + 0.5) / float2(size) * 2.0 - 1.0;
  RayDesc ray;
  ray.Origin = cam.origin.xyz;
  ray.Direction = normalize(cam.forward.xyz + uv.x * cam.right.xyz - uv.y * cam.up.xyz);
  ray.TMin = 0.001;
  ray.TMax = 10000.0;
  Payload p;
  p.color = float3(0.0);
  TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, p);
  img[id] = float4(toSrgb(p.color), 1.0);
}

[shader("miss")]
void miss(inout Payload p) {
  float t = 0.5 * (WorldRayDirection().y + 1.0);
  p.color = lerp(float3(1.0), float3(0.4, 0.6, 1.0), t);
}

[shader("miss")]
void shadowMiss(inout ShadowPayload s) { s.visible = 1; }

[shader("anyhit")]
void ahit(inout Payload p, in BuiltInTriangleIntersectionAttributes attr) {
  if (baseColor(triangleId(), attr.barycentrics).a < 0.5) IgnoreHit();
}

[shader("anyhit")]
void ahitShadow(inout ShadowPayload s, in BuiltInTriangleIntersectionAttributes attr) {
  if (baseColor(triangleId(), attr.barycentrics).a < 0.5) IgnoreHit();
}

[shader("closesthit")]
void chit(inout Payload p, in BuiltInTriangleIntersectionAttributes attr) {
  uint tri = triangleId();
  float4 nw = normals[tri];
  // Geometry may be one-sided or back-facing; shade the side the ray hit.
  float3 N = dot(nw.xyz, WorldRayDirection()) > 0.0 ? -nw.xyz : nw.xyz;
  float3 P = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
  float3 L = cam.sun.xyz;
  float3 albedo = baseColor(tri, attr.barycentrics).rgb;
  RayDesc sray;
  sray.Origin = P + N * 1e-3;
  sray.Direction = L;
  sray.TMin = 0.0;
  sray.TMax = 10000.0;
  ShadowPayload s;
  s.visible = 0;
  TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 1, 0, 1, sray, s);
  float ndl = max(dot(N, L), 0.0);
  p.color = albedo * (0.15 + 0.85 * ndl * float(s.visible));
}
