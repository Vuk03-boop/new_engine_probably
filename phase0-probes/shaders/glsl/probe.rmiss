#version 460
#extension GL_EXT_ray_tracing : require
layout(location = 0) rayPayloadInEXT vec3 color;
void main() {
  float t = 0.5 * (gl_WorldRayDirectionEXT.y + 1.0);
  color = mix(vec3(1.0), vec3(0.4, 0.6, 1.0), t);
}
