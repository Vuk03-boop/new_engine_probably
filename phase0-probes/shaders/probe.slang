// Same behaviour as glsl/*.r* — keep the two in sync.
struct Payload { float3 color; };
struct ShadowPayload { uint visible; };

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> img;
[[vk::binding(2, 0)]] StructuredBuffer<float4> normals;

// right is pre-scaled by aspect * tan(fov_y / 2), up by tan(fov_y / 2); forward is not normalized.
// sun.xyz = normalized direction towards the sun.
struct Camera { float4 origin; float4 forward; float4 right; float4 up; float4 sun; };
[[vk::push_constant]] ConstantBuffer<Camera> cam;

// Only varies the pipeline's specialization data so the driver cannot reuse a cached compile
// (window_probe cold-compile test). 0xFFFFFFFF is never used, so shading is unchanged.
[vk::constant_id(0)] const uint kCompileSeed = 0;

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
  TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, ray, p);
  if (kCompileSeed == 0xFFFFFFFFu) p.color *= 0.5;
  img[id] = float4(p.color, 1.0);
}

[shader("miss")]
void miss(inout Payload p) {
  float t = 0.5 * (WorldRayDirection().y + 1.0);
  p.color = lerp(float3(1.0), float3(0.4, 0.6, 1.0), t);
}

[shader("miss")]
void shadowMiss(inout ShadowPayload s) { s.visible = 1; }

[shader("closesthit")]
void chit(inout Payload p, in BuiltInTriangleIntersectionAttributes attr) {
  float4 nw = normals[PrimitiveIndex()];
  // Geometry may be one-sided or back-facing; shade the side the ray hit.
  float3 N = dot(nw.xyz, WorldRayDirection()) > 0.0 ? -nw.xyz : nw.xyz;
  float3 P = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
  float3 L = cam.sun.xyz;
  float3 albedo = nw.w > 0.5 ? float3(0.8, 0.3, 0.2) : float3(0.7);
  RayDesc sray;
  sray.Origin = P + N * 1e-3;
  sray.Direction = L;
  sray.TMin = 0.0;
  sray.TMax = 10000.0;
  ShadowPayload s;
  s.visible = 0;
  TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
           0xFF, 0, 0, 1, sray, s);
  float ndl = max(dot(N, L), 0.0);
  p.color = albedo * (0.15 + 0.85 * ndl * float(s.visible));
}
