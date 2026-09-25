#version 460
#extension GL_EXT_ray_tracing : require
layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 2, std430) readonly buffer Normals { vec4 n[]; } normals;
layout(location = 0) rayPayloadInEXT vec3 color;
layout(location = 1) rayPayloadEXT uint visible;
hitAttributeEXT vec2 bary;
layout(push_constant) uniform Camera { vec4 origin; vec4 forward; vec4 right; vec4 up; vec4 sun; } cam;
void main() {
  vec4 nw = normals.n[gl_PrimitiveID];
  // Geometry may be one-sided or back-facing; shade the side the ray hit.
  vec3 N = dot(nw.xyz, gl_WorldRayDirectionEXT) > 0.0 ? -nw.xyz : nw.xyz;
  vec3 P = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  vec3 L = cam.sun.xyz;
  vec3 albedo = nw.w > 0.5 ? vec3(0.8, 0.3, 0.2) : vec3(0.7);
  visible = 0u;
  traceRayEXT(tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              0xFF, 0, 0, 1, P + N * 1e-3, 0.0, L, 10000.0, 1);
  float ndl = max(dot(N, L), 0.0);
  color = albedo * (0.15 + 0.85 * ndl * float(visible));
}
