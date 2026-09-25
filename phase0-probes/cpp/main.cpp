// Phase 0 RT probe (C++). Throwaway: checks device selection, AS builds, RT pipeline creation
// from pre-compiled SPIR-V, traced images, timestamps and memory. Not engine code.
//
// usage: rt_probe <shader_dir> <out.ppm> [--size W H] [--fbx scene.fbx] [--textures-out list.txt]
// shader_dir must contain rgen.spv miss.spv shadow.spv chit.spv (see ../shaders/build_shaders.bat).
// Must stay behaviourally identical to ../rust/src/main.rs (same scene, camera, math order, output).

#include <vulkan/vulkan.h>

#include "ufbx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define VK(x)                                                                        \
  do {                                                                               \
    VkResult r_ = (x);                                                               \
    if (r_ != VK_SUCCESS) {                                                          \
      std::fprintf(stderr, "FAIL %s -> %d (%s:%d)\n", #x, (int)r_, __FILE__, __LINE__); \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

static int g_validation_errors = 0, g_validation_warnings = 0;

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev,
                                               VkDebugUtilsMessageTypeFlagsEXT,
                                               const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
  if (sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) g_validation_errors++;
  else if (sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) g_validation_warnings++;
  std::fprintf(stderr, "[validation] %s\n", d->pMessage);
  return VK_FALSE;
}

[[noreturn]] static void fail(const std::string& msg) {
  std::fprintf(stderr, "FAIL %s\n", msg.c_str());
  std::exit(1);
}

static std::vector<uint32_t> read_spv(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) fail("cannot open " + path);
  size_t n = (size_t)f.tellg();
  std::vector<uint32_t> w(n / 4);
  f.seekg(0);
  f.read((char*)w.data(), (std::streamsize)n);
  return w;
}

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// ---------------------------------------------------------------- scene
// Triangles in world space. nrm[i] = (geometric normal of triangle i, 1 for box / 0 otherwise).
struct Scene {
  std::vector<float> pos;
  std::vector<uint32_t> idx;
  std::vector<float> nrm;
  float cam[20] = {};  // push constants: origin, forward, right (scaled), up (scaled), sun dir; w unused
  std::string sun_source = "default";
  size_t lights = 0;
  std::string name, camera_name;
  size_t meshes = 0, nodes = 0, cameras = 0, textures = 0;
  struct Range { uint32_t first_tri; std::string node, material; };
  std::vector<Range> ranges;  // diagnostics only (PROBE_PICK)
};

static void add_quad(Scene& s, const float p[4][3], const float n[3], float flag) {
  uint32_t base = (uint32_t)(s.pos.size() / 3);
  for (int i = 0; i < 4; ++i) s.pos.insert(s.pos.end(), {p[i][0], p[i][1], p[i][2]});
  s.idx.insert(s.idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
  for (int t = 0; t < 2; ++t) s.nrm.insert(s.nrm.end(), {n[0], n[1], n[2], flag});
}

static void add_box(Scene& s, float x0, float y0, float z0, float x1, float y1, float z1) {
  const float f[6][4][3] = {
      {{x1, y0, z0}, {x1, y1, z0}, {x1, y1, z1}, {x1, y0, z1}},  // +x
      {{x0, y0, z1}, {x0, y1, z1}, {x0, y1, z0}, {x0, y0, z0}},  // -x
      {{x0, y1, z0}, {x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}},  // +y
      {{x0, y0, z1}, {x0, y0, z0}, {x1, y0, z0}, {x1, y0, z1}},  // -y
      {{x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}},  // +z
      {{x1, y0, z0}, {x0, y0, z0}, {x0, y1, z0}, {x1, y1, z0}},  // -z
  };
  const float n[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (int i = 0; i < 6; ++i) add_quad(s, f[i], n[i], 1.0f);
}

// Direction towards the sun used when a scene has no directional light.
static void set_default_sun(Scene& s) {
  const double l = std::sqrt(0.5 * 0.5 + 1.0 * 1.0 + 0.3 * 0.3);
  s.cam[16] = (float)(0.5 / l);
  s.cam[17] = (float)(1.0 / l);
  s.cam[18] = (float)(0.3 / l);
}

static Scene make_small_scene(uint32_t w, uint32_t h) {
  Scene s;
  s.name = "small";
  const float g[4][3] = {{-10, 0, -10}, {-10, 0, 10}, {10, 0, 10}, {10, 0, -10}};
  const float up[3] = {0, 1, 0};
  add_quad(s, g, up, 0.0f);
  add_box(s, -0.5f, 0.0f, -0.5f, 0.5f, 1.0f, 0.5f);
  add_box(s, 1.0f, 0.0f, -2.0f, 2.0f, 2.5f, -1.0f);
  const float aspect = (float)w / (float)h;
  const float cam[16] = {0, 1.5f, 5, 0, 0, -0.15f, -1, 0, aspect * 0.6f, 0, 0, 0, 0, 0.6f, 0, 0};
  std::memcpy(s.cam, cam, sizeof(cam));
  set_default_sun(s);
  return s;
}

// Every mesh instance flattened into one world-space triangle list, meters, right-handed Y-up.
// Camera: first FBX camera at its default pose; falls back to looking at the scene bounds.
static Scene load_fbx(const char* path, uint32_t w, uint32_t h, const char* textures_out) {
  ufbx_load_opts o{};
  o.target_axes = ufbx_axes_right_handed_y_up;
  o.target_camera_axes = ufbx_axes_right_handed_y_up;
  o.target_unit_meters = 1.0;
  o.space_conversion = UFBX_SPACE_CONVERSION_MODIFY_GEOMETRY;
  ufbx_error err;
  ufbx_scene* sc = ufbx_load_file(path, &o, &err);
  if (!sc) fail("ufbx: " + std::string(err.description.data, err.description.length));

  Scene s;
  s.name = "fbx";
  s.meshes = sc->meshes.count;
  s.nodes = sc->nodes.count;
  s.cameras = sc->cameras.count;
  s.textures = sc->textures.count;
  std::vector<uint32_t> tri;
  for (size_t ni = 0; ni < sc->nodes.count; ++ni) {
    const ufbx_node* node = sc->nodes.data[ni];
    const ufbx_mesh* mesh = node->mesh;
    if (!mesh) continue;
    const uint32_t base = (uint32_t)(s.pos.size() / 3);
    s.ranges.push_back({(uint32_t)(s.idx.size() / 3), std::string(node->name.data, node->name.length),
                        mesh->materials.count ? std::string(mesh->materials.data[0]->name.data, mesh->materials.data[0]->name.length) : ""});
    for (size_t v = 0; v < mesh->vertices.count; ++v) {
      ufbx_vec3 p = ufbx_transform_position(&node->geometry_to_world, mesh->vertices.data[v]);
      s.pos.insert(s.pos.end(), {(float)p.x, (float)p.y, (float)p.z});
    }
    tri.resize(mesh->max_face_triangles * 3);
    for (size_t f = 0; f < mesh->faces.count; ++f) {
      uint32_t nt = ufbx_triangulate_face(tri.data(), tri.size(), mesh, mesh->faces.data[f]);
      for (uint32_t t = 0; t < nt; ++t) {
        uint32_t ia = base + mesh->vertex_indices.data[tri[t * 3 + 0]];
        uint32_t ib = base + mesh->vertex_indices.data[tri[t * 3 + 1]];
        uint32_t ic = base + mesh->vertex_indices.data[tri[t * 3 + 2]];
        s.idx.insert(s.idx.end(), {ia, ib, ic});
        const float* a = &s.pos[ia * 3];
        const float* b = &s.pos[ib * 3];
        const float* c = &s.pos[ic * 3];
        float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        float e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 1e-20f) s.nrm.insert(s.nrm.end(), {n[0] / len, n[1] / len, n[2] / len, 0.0f});
        else s.nrm.insert(s.nrm.end(), {0.0f, 1.0f, 0.0f, 0.0f});  // degenerate triangle
      }
    }
  }

  const double aspect = (double)w / (double)h;
  const ufbx_camera* camera = sc->cameras.count ? sc->cameras.data[0] : nullptr;
  if (camera && camera->instances.count && camera->projection_mode == UFBX_PROJECTION_MODE_PERSPECTIVE) {
    const ufbx_matrix& m = camera->instances.data[0]->node_to_world;
    auto norm = [](ufbx_vec3 v) {
      double l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
      return ufbx_vec3{v.x / l, v.y / l, v.z / l};
    };
    ufbx_vec3 r = norm(m.cols[0]), u = norm(m.cols[1]), f = norm(m.cols[2]);
    const double ty = camera->field_of_view_tan.y;
    const float cam[16] = {(float)m.cols[3].x, (float)m.cols[3].y, (float)m.cols[3].z, 0,
                           (float)-f.x, (float)-f.y, (float)-f.z, 0,
                           (float)(r.x * aspect * ty), (float)(r.y * aspect * ty), (float)(r.z * aspect * ty), 0,
                           (float)(u.x * ty), (float)(u.y * ty), (float)(u.z * ty), 0};
    std::memcpy(s.cam, cam, sizeof(cam));
    s.camera_name.assign(camera->name.data, camera->name.length);
  } else {
    fail("fbx has no perspective camera");
  }

  // Sun: first directional light; ufbx gives its aim direction in node space, we need the direction towards it.
  s.lights = sc->lights.count;
  set_default_sun(s);
  for (size_t i = 0; i < sc->lights.count; ++i) {
    const ufbx_light* light = sc->lights.data[i];
    if (light->type != UFBX_LIGHT_DIRECTIONAL || !light->instances.count) continue;
    ufbx_vec3 d = ufbx_transform_direction(&light->instances.data[0]->node_to_world, light->local_direction);
    const double l = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    s.cam[16] = (float)(-d.x / l);
    s.cam[17] = (float)(-d.y / l);
    s.cam[18] = (float)(-d.z / l);
    s.sun_source.assign(light->name.data, light->name.length);
    break;
  }

  if (textures_out) {
    std::ofstream t(textures_out);
    for (size_t i = 0; i < sc->textures.count; ++i) {
      const ufbx_string& fn = sc->textures.data[i]->filename;
      t << std::string(fn.data, fn.length) << "\n";
    }
  }
  ufbx_free_scene(sc);
  return s;
}

// Diagnostic: brute-force the primary ray of pixel (px, py) on the CPU (same camera math as probe.rgen)
// and report the nearest triangle, its distance and the FBX node/material it came from.
static void cpu_pick(const Scene& s, uint32_t w, uint32_t h, uint32_t px, uint32_t py) {
  const float u = ((float)px + 0.5f) / (float)w * 2.0f - 1.0f, v = ((float)py + 0.5f) / (float)h * 2.0f - 1.0f;
  double d[3];
  for (int k = 0; k < 3; ++k) d[k] = s.cam[4 + k] + u * s.cam[8 + k] - v * s.cam[12 + k];
  const double o[3] = {s.cam[0], s.cam[1], s.cam[2]};
  const double dl = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  double best = 1e30;
  int64_t best_tri = -1;
  for (size_t t = 0; t < s.idx.size() / 3; ++t) {
    const float* a = &s.pos[s.idx[t * 3] * 3];
    const float* b = &s.pos[s.idx[t * 3 + 1] * 3];
    const float* c = &s.pos[s.idx[t * 3 + 2] * 3];
    double e1[3], e2[3], pv[3], tv[3], qv[3];
    for (int k = 0; k < 3; ++k) { e1[k] = b[k] - a[k]; e2[k] = c[k] - a[k]; tv[k] = o[k] - a[k]; }
    pv[0] = d[1] * e2[2] - d[2] * e2[1]; pv[1] = d[2] * e2[0] - d[0] * e2[2]; pv[2] = d[0] * e2[1] - d[1] * e2[0];
    double det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
    if (std::fabs(det) < 1e-18) continue;
    double inv = 1.0 / det, bu = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
    if (bu < 0 || bu > 1) continue;
    qv[0] = tv[1] * e1[2] - tv[2] * e1[1]; qv[1] = tv[2] * e1[0] - tv[0] * e1[2]; qv[2] = tv[0] * e1[1] - tv[1] * e1[0];
    double bv = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) * inv;
    if (bv < 0 || bu + bv > 1) continue;
    double tt = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
    if (tt * dl > 0.001 && tt < best) { best = tt; best_tri = (int64_t)t; }  // tmin as in probe.rgen
  }
  if (best_tri < 0) { std::fprintf(stderr, "[pick] (%u,%u) miss\n", px, py); return; }
  const Scene::Range* r = nullptr;
  for (const auto& rg : s.ranges) if (rg.first_tri <= (uint32_t)best_tri) r = &rg;
  const float* n = &s.nrm[best_tri * 4];
  std::fprintf(stderr, "[pick] (%u,%u) tri=%lld dist=%.3f m node='%s' material='%s' normal=(%.2f,%.2f,%.2f)\n", px, py,
               (long long)best_tri, best * dl, r ? r->node.c_str() : "?", r ? r->material.c_str() : "?", n[0], n[1], n[2]);
}

// ---------------------------------------------------------------- vulkan helpers
struct Ctx {
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice dev = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties mem{};
};

struct Buffer { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkDeviceAddress addr = 0; void* map = nullptr; VkDeviceSize size = 0; };

static uint32_t find_mem(const Ctx& c, uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < c.mem.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (c.mem.memoryTypes[i].propertyFlags & want) == want) return i;
  fail("no memory type");
}

static Buffer make_buffer(const Ctx& c, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                          const void* data = nullptr) {
  Buffer b;
  b.size = size;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VK(vkCreateBuffer(c.dev, &bi, nullptr, &b.buf));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(c.dev, b.buf, &req);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &fi;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = find_mem(c, req.memoryTypeBits, props);
  VK(vkAllocateMemory(c.dev, &ai, nullptr, &b.mem));
  VK(vkBindBufferMemory(c.dev, b.buf, b.mem, 0));
  if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    VK(vkMapMemory(c.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    if (data) std::memcpy(b.map, data, (size_t)size);
  }
  VkBufferDeviceAddressInfo da{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  da.buffer = b.buf;
  b.addr = vkGetBufferDeviceAddress(c.dev, &da);
  return b;
}

static void destroy_buffer(const Ctx& c, Buffer& b) {
  vkDestroyBuffer(c.dev, b.buf, nullptr);
  vkFreeMemory(c.dev, b.mem, nullptr);
  b = {};
}

// Sum of heapUsage over device-local heaps (VK_EXT_memory_budget), in bytes; 0 if unavailable.
static void device_local_usage(const Ctx& c, bool have_budget, uint64_t& usage, uint64_t& budget) {
  usage = budget = 0;
  if (!have_budget) return;
  VkPhysicalDeviceMemoryBudgetPropertiesEXT mb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  mp.pNext = &mb;
  vkGetPhysicalDeviceMemoryProperties2(c.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount; ++i)
    if (mp.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      usage += mb.heapUsage[i];
      budget += mb.heapBudget[i];
    }
}

#define LOAD(name) auto name = (PFN_##name)vkGetDeviceProcAddr(dev, #name); if (!name) fail("load " #name);

constexpr uint32_t kWarmTraces = 5;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: rt_probe <shader_dir> <out.ppm> [--size W H] [--fbx scene.fbx] [--textures-out list.txt]\n");
    return 2;
  }
  const std::string shader_dir = argv[1], out_path = argv[2];
  uint32_t W = 1280, H = 720;
  const char* fbx_path = nullptr;
  const char* textures_out = nullptr;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--size" && i + 2 < argc) { W = (uint32_t)std::atoi(argv[i + 1]); H = (uint32_t)std::atoi(argv[i + 2]); i += 2; }
    else if (a == "--fbx" && i + 1 < argc) fbx_path = argv[++i];
    else if (a == "--textures-out" && i + 1 < argc) textures_out = argv[++i];
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  auto t_start = clk::now();

  // ---- scene (CPU)
  Scene scene = fbx_path ? load_fbx(fbx_path, W, H, textures_out) : make_small_scene(W, H);
  auto t_scene = clk::now();
  if (const char* pick = std::getenv("PROBE_PICK")) {  // "x,y;x,y;..." diagnostic, then exit
    std::fprintf(stderr, "[pick] camera origin (%.3f %.3f %.3f) forward (%.3f %.3f %.3f) sun (%.3f %.3f %.3f) from '%s'\n",
                 scene.cam[0], scene.cam[1], scene.cam[2], scene.cam[4], scene.cam[5], scene.cam[6], scene.cam[16],
                 scene.cam[17], scene.cam[18], scene.sun_source.c_str());
    for (const char* q = pick; *q;) {
      unsigned x = 0, y = 0;
      if (std::sscanf(q, "%u,%u", &x, &y) != 2) break;
      cpu_pick(scene, W, H, x, y);
      q = std::strchr(q, ';');
      if (!q) break;
      ++q;
    }
    return 0;
  }
  const uint32_t tri_count = (uint32_t)(scene.idx.size() / 3);
  const uint32_t vert_count = (uint32_t)(scene.pos.size() / 3);

  // ---- instance (+ validation if present)
  uint32_t nl = 0;
  vkEnumerateInstanceLayerProperties(&nl, nullptr);
  std::vector<VkLayerProperties> layers(nl);
  vkEnumerateInstanceLayerProperties(&nl, layers.data());
  bool have_validation = false;
  for (auto& l : layers) have_validation |= std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0;
  if (std::getenv("PROBE_NO_VALIDATION")) have_validation = false;
  const char* layer_names[] = {"VK_LAYER_KHRONOS_validation"};
  const char* inst_exts[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "rt_probe_cpp";
  app.apiVersion = VK_API_VERSION_1_3;
  VkDebugUtilsMessengerCreateInfoEXT dbg{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  dbg.pfnUserCallback = debug_cb;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  if (have_validation) {
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = layer_names;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = inst_exts;
    ici.pNext = &dbg;
  }
  VkInstance inst;
  VK(vkCreateInstance(&ici, nullptr, &inst));
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  if (have_validation) {
    auto create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
    VK(create(inst, &dbg, nullptr, &messenger));
  }

  // ---- pick device by capability, never by index
  const char* required_exts[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                 VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
  uint32_t nd = 0;
  vkEnumeratePhysicalDevices(inst, &nd, nullptr);
  std::vector<VkPhysicalDevice> devs(nd);
  vkEnumeratePhysicalDevices(inst, &nd, devs.data());
  Ctx c;
  int best_score = -1;
  bool have_budget = false;
  for (auto p : devs) {
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(p, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ex(ne);
    vkEnumerateDeviceExtensionProperties(p, nullptr, &ne, ex.data());
    auto has = [&](const char* want) {
      for (auto& e : ex) if (std::strcmp(e.extensionName, want) == 0) return true;
      return false;
    };
    if (!std::all_of(std::begin(required_exts), std::end(required_exts), has)) continue;
    VkPhysicalDeviceProperties pr;
    vkGetPhysicalDeviceProperties(p, &pr);
    int score = pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
    if (score > best_score) { best_score = score; c.phys = p; have_budget = has(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME); }
  }
  if (!c.phys) fail("no ray-tracing capable device");

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  rtp.pNext = &asp;
  VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &rtp;
  vkGetPhysicalDeviceProperties2(c.phys, &props2);
  vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mem);
  const auto& pr = props2.properties;
  if (rtp.maxRayRecursionDepth < 2) fail("maxRayRecursionDepth < 2");

  uint32_t nq = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nq);
  vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, qf.data());
  uint32_t qfi = UINT32_MAX;
  for (uint32_t i = 0; i < nq; ++i)
    if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits > 0) { qfi = i; break; }
  if (qfi == UINT32_MAX) fail("no compute queue with timestamps");

  // ---- device
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.bufferDeviceAddress = VK_TRUE;
  VkPhysicalDeviceAccelerationStructureFeaturesKHR fas{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  fas.accelerationStructure = VK_TRUE;
  fas.pNext = &f12;
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR frt{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  frt.rayTracingPipeline = VK_TRUE;
  frt.pNext = &fas;
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfi;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  std::vector<const char*> dev_exts(std::begin(required_exts), std::end(required_exts));
  if (have_budget) dev_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &frt;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t)dev_exts.size();
  dci.ppEnabledExtensionNames = dev_exts.data();
  VK(vkCreateDevice(c.phys, &dci, nullptr, &c.dev));
  VkDevice dev = c.dev;
  VkQueue queue;
  vkGetDeviceQueue(dev, qfi, 0, &queue);
  auto t_device = clk::now();
  uint64_t vram0, budget0;
  device_local_usage(c, have_budget, vram0, budget0);

  LOAD(vkCreateAccelerationStructureKHR);
  LOAD(vkDestroyAccelerationStructureKHR);
  LOAD(vkGetAccelerationStructureBuildSizesKHR);
  LOAD(vkCmdBuildAccelerationStructuresKHR);
  LOAD(vkGetAccelerationStructureDeviceAddressKHR);
  LOAD(vkCmdWriteAccelerationStructuresPropertiesKHR);
  LOAD(vkCreateRayTracingPipelinesKHR);
  LOAD(vkGetRayTracingShaderGroupHandlesKHR);
  LOAD(vkCmdTraceRaysKHR);

  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.queueFamilyIndex = qfi;
  VkCommandPool pool;
  VK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
  // timestamps: [0,1] AS build, then (start, end) for 1 cold + kWarmTraces warm traces
  const uint32_t n_ts = 2 + 2 * (1 + kWarmTraces);
  VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpi.queryCount = n_ts;
  VkQueryPool qpool;
  VK(vkCreateQueryPool(dev, &qpi, nullptr, &qpool));
  VkQueryPoolCreateInfo cqpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  cqpi.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
  cqpi.queryCount = 1;
  VkQueryPool cqpool;
  VK(vkCreateQueryPool(dev, &cqpi, nullptr, &cqpool));

  auto run = [&](auto&& rec) {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK(vkBeginCommandBuffer(cb, &bi));
    rec(cb);
    VK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK(vkCreateFence(dev, &fci, nullptr, &fence));
    VK(vkQueueSubmit(queue, 1, &si, fence));
    VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(dev, fence, nullptr);
    vkFreeCommandBuffers(dev, pool, 1, &cb);
  };

  const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const VkMemoryPropertyFlags local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  // Device-local buffer filled through a temporary host-visible staging buffer.
  auto upload = [&](const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer staging = make_buffer(c, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, data);
    Buffer dst = make_buffer(c, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, local);
    run([&](VkCommandBuffer cb) {
      VkBufferCopy region{0, 0, size};
      vkCmdCopyBuffer(cb, staging.buf, dst.buf, 1, &region);
    });
    destroy_buffer(c, staging);
    return dst;
  };

  // ---- geometry upload
  auto t_up0 = clk::now();
  const VkBufferUsageFlags as_input = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  Buffer vbuf = upload(scene.pos.data(), scene.pos.size() * 4, as_input);
  Buffer ibuf = upload(scene.idx.data(), scene.idx.size() * 4, as_input);
  Buffer nbuf = upload(scene.nrm.data(), scene.nrm.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  auto t_up1 = clk::now();

  // ---- BLAS
  VkAccelerationStructureGeometryKHR geo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  auto& tri = geo.geometry.triangles;
  tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  tri.vertexData.deviceAddress = vbuf.addr;
  tri.vertexStride = 12;
  tri.maxVertex = vert_count - 1;
  tri.indexType = VK_INDEX_TYPE_UINT32;
  tri.indexData.deviceAddress = ibuf.addr;

  auto build_as = [&](VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags,
                      VkAccelerationStructureGeometryKHR& g, uint32_t prim_count, Buffer& as_buf,
                      VkAccelerationStructureKHR& as, Buffer& scratch) {
    VkAccelerationStructureBuildGeometryInfoKHR bgi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bgi.type = type;
    bgi.flags = flags;
    bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bgi.geometryCount = 1;
    bgi.pGeometries = &g;
    VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &prim_count, &sz);
    as_buf = make_buffer(c, sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, local);
    VkAccelerationStructureCreateInfoKHR aci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    aci.buffer = as_buf.buf;
    aci.size = sz.accelerationStructureSize;
    aci.type = type;
    VK(vkCreateAccelerationStructureKHR(dev, &aci, nullptr, &as));
    const uint64_t sa = asp.minAccelerationStructureScratchOffsetAlignment;
    scratch = make_buffer(c, sz.buildScratchSize + sa, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, local);
    bgi.dstAccelerationStructure = as;
    bgi.scratchData.deviceAddress = align_up(scratch.addr, sa);
    return std::make_pair(bgi, sz);
  };

  Buffer blas_buf, blas_scratch;
  VkAccelerationStructureKHR blas;
  auto [blas_info, blas_sz] = build_as(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                       VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                           VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR,
                                       geo, tri_count, blas_buf, blas, blas_scratch);
  VkAccelerationStructureDeviceAddressInfoKHR bai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  bai.accelerationStructure = blas;
  VkDeviceAddress blas_addr = vkGetAccelerationStructureDeviceAddressKHR(dev, &bai);

  VkAccelerationStructureInstanceKHR instance{};
  instance.transform.matrix[0][0] = instance.transform.matrix[1][1] = instance.transform.matrix[2][2] = 1.0f;
  instance.mask = 0xFF;
  instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  instance.accelerationStructureReference = blas_addr;
  Buffer inst_buf = upload(&instance, sizeof(instance), as_input);

  VkAccelerationStructureGeometryKHR tgeo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  tgeo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  tgeo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  tgeo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  tgeo.geometry.instances.data.deviceAddress = inst_buf.addr;
  Buffer tlas_buf, tlas_scratch;
  VkAccelerationStructureKHR tlas;
  auto [tlas_info, tlas_sz] = build_as(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                       VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, tgeo, 1, tlas_buf,
                                       tlas, tlas_scratch);

  VkAccelerationStructureBuildRangeInfoKHR blas_range{tri_count, 0, 0, 0}, tlas_range{1, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR* blas_ranges = &blas_range;
  const VkAccelerationStructureBuildRangeInfoKHR* tlas_ranges = &tlas_range;
  run([&](VkCommandBuffer cb) {
    vkCmdResetQueryPool(cb, qpool, 0, n_ts);
    vkCmdResetQueryPool(cb, cqpool, 0, 1);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    vkCmdBuildAccelerationStructuresKHR(cb, 1, &blas_info, &blas_ranges);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &mb, 0, nullptr, 0, nullptr);
    vkCmdWriteAccelerationStructuresPropertiesKHR(cb, 1, &blas, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                                                  cqpool, 0);
    vkCmdBuildAccelerationStructuresKHR(cb, 1, &tlas_info, &tlas_ranges);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
  });
  uint64_t blas_compacted = 0;
  VK(vkGetQueryPoolResults(dev, cqpool, 0, 1, sizeof(blas_compacted), &blas_compacted, 8,
                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  // Scratch is only needed during the build; free it before measuring resident memory.
  destroy_buffer(c, blas_scratch);
  destroy_buffer(c, tlas_scratch);
  auto t_as = clk::now();

  // ---- output image + readback
  VkImageCreateInfo imci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imci.imageType = VK_IMAGE_TYPE_2D;
  imci.format = VK_FORMAT_R8G8B8A8_UNORM;
  imci.extent = {W, H, 1};
  imci.mipLevels = imci.arrayLayers = 1;
  imci.samples = VK_SAMPLE_COUNT_1_BIT;
  imci.tiling = VK_IMAGE_TILING_OPTIMAL;
  imci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  VkImage image;
  VK(vkCreateImage(dev, &imci, nullptr, &image));
  VkMemoryRequirements ireq;
  vkGetImageMemoryRequirements(dev, image, &ireq);
  VkMemoryAllocateInfo iai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  iai.allocationSize = ireq.size;
  iai.memoryTypeIndex = find_mem(c, ireq.memoryTypeBits, local);
  VkDeviceMemory image_mem;
  VK(vkAllocateMemory(dev, &iai, nullptr, &image_mem));
  VK(vkBindImageMemory(dev, image, image_mem, 0));
  VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  ivci.image = image;
  ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  ivci.format = imci.format;
  ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view;
  VK(vkCreateImageView(dev, &ivci, nullptr, &view));
  Buffer readback = make_buffer(c, (VkDeviceSize)W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                host | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

  // ---- descriptors
  VkDescriptorSetLayoutBinding binds[3] = {
      {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr},
  };
  VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dlci.bindingCount = 3;
  dlci.pBindings = binds;
  VkDescriptorSetLayout dsl;
  VK(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &dsl));
  VkDescriptorPoolSize psz[3] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 3;
  dpci.pPoolSizes = psz;
  VkDescriptorPool dpool;
  VK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dpool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet dset;
  VK(vkAllocateDescriptorSets(dev, &dsai, &dset));
  VkWriteDescriptorSetAccelerationStructureKHR was{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  was.accelerationStructureCount = 1;
  was.pAccelerationStructures = &tlas;
  VkDescriptorImageInfo dii{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorBufferInfo dbi{nbuf.buf, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet w[3]{};
  for (int i = 0; i < 3; ++i) {
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = dset;
    w[i].dstBinding = (uint32_t)i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = binds[i].descriptorType;
  }
  w[0].pNext = &was;
  w[1].pImageInfo = &dii;
  w[2].pBufferInfo = &dbi;
  vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);

  // ---- pipeline
  auto t_pipe0 = clk::now();
  const char* files[4] = {"rgen.spv", "miss.spv", "shadow.spv", "chit.spv"};
  const VkShaderStageFlagBits stages[4] = {VK_SHADER_STAGE_RAYGEN_BIT_KHR, VK_SHADER_STAGE_MISS_BIT_KHR,
                                           VK_SHADER_STAGE_MISS_BIT_KHR, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR};
  VkShaderModule modules[4];
  VkPipelineShaderStageCreateInfo ssci[4]{};
  for (int i = 0; i < 4; ++i) {
    auto code = read_spv(shader_dir + "/" + files[i]);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size() * 4;
    smci.pCode = code.data();
    VK(vkCreateShaderModule(dev, &smci, nullptr, &modules[i]));
    ssci[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssci[i].stage = stages[i];
    ssci[i].module = modules[i];
    ssci[i].pName = "main";
  }
  VkRayTracingShaderGroupCreateInfoKHR groups[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    groups[i].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[i].type = i == 3 ? VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR : VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[i].generalShader = i == 3 ? VK_SHADER_UNUSED_KHR : i;
    groups[i].closestHitShader = i == 3 ? 3 : VK_SHADER_UNUSED_KHR;
    groups[i].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[i].intersectionShader = VK_SHADER_UNUSED_KHR;
  }
  VkPushConstantRange pcr{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(scene.cam)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout layout;
  VK(vkCreatePipelineLayout(dev, &plci, nullptr, &layout));
  VkRayTracingPipelineCreateInfoKHR rpci{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rpci.stageCount = 4;
  rpci.pStages = ssci;
  rpci.groupCount = 4;
  rpci.pGroups = groups;
  rpci.maxPipelineRayRecursionDepth = 2;
  rpci.layout = layout;
  VkPipeline pipeline;
  VK(vkCreateRayTracingPipelinesKHR(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rpci, nullptr, &pipeline));
  auto t_pipe1 = clk::now();

  // ---- shader binding table: [raygen][miss, shadow miss][hit], each region base-aligned
  const uint32_t hsize = rtp.shaderGroupHandleSize;
  const uint64_t hstride = align_up(hsize, rtp.shaderGroupHandleAlignment);
  const uint64_t base = rtp.shaderGroupBaseAlignment;
  const uint64_t rgen_size = align_up(hstride, base), miss_size = align_up(2 * hstride, base), hit_size = align_up(hstride, base);
  std::vector<uint8_t> handles(4 * hsize);
  VK(vkGetRayTracingShaderGroupHandlesKHR(dev, pipeline, 0, 4, handles.size(), handles.data()));
  Buffer sbt = make_buffer(c, rgen_size + miss_size + hit_size + base, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR, host);
  const uint64_t sbt_base = align_up(sbt.addr, base);
  uint8_t* sbt_ptr = (uint8_t*)sbt.map + (sbt_base - sbt.addr);
  std::memcpy(sbt_ptr, &handles[0], hsize);
  std::memcpy(sbt_ptr + rgen_size, &handles[hsize], hsize);
  std::memcpy(sbt_ptr + rgen_size + hstride, &handles[2 * hsize], hsize);
  std::memcpy(sbt_ptr + rgen_size + miss_size, &handles[3 * hsize], hsize);
  VkStridedDeviceAddressRegionKHR r_rgen{sbt_base, rgen_size, rgen_size};
  VkStridedDeviceAddressRegionKHR r_miss{sbt_base + rgen_size, hstride, miss_size};
  VkStridedDeviceAddressRegionKHR r_hit{sbt_base + rgen_size + miss_size, hstride, hit_size};
  VkStridedDeviceAddressRegionKHR r_call{};

  uint64_t vram1, budget1;
  device_local_usage(c, have_budget, vram1, budget1);

  // ---- trace (1 cold + kWarmTraces warm, identical output) + readback
  run([&](VkCommandBuffer cb) {
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = 0;
    ib.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = image;
    ib.subresourceRange = ivci.subresourceRange;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0,
                         nullptr, 0, nullptr, 1, &ib);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout, 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(scene.cam), scene.cam);
    for (uint32_t k = 0; k < 1 + kWarmTraces; ++k) {
      if (k > 0) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                             VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &mb, 0, nullptr, 0, nullptr);
      }
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 2 + 2 * k);
      vkCmdTraceRaysKHR(cb, &r_rgen, &r_miss, &r_hit, &r_call, W, H, 1);
      vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 3 + 2 * k);
    }
    ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &ib);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &copy);
    VkBufferMemoryBarrier hb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    hb.srcQueueFamilyIndex = hb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hb.buffer = readback.buf;
    hb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hb, 0, nullptr);
  });

  std::vector<uint64_t> ts(n_ts);
  VK(vkGetQueryPoolResults(dev, qpool, 0, n_ts, ts.size() * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  const double tick_ms = pr.limits.timestampPeriod / 1e6;
  std::vector<double> warm;
  for (uint32_t k = 1; k <= kWarmTraces; ++k) warm.push_back((ts[3 + 2 * k] - ts[2 + 2 * k]) * tick_ms);
  std::sort(warm.begin(), warm.end());

  // ---- write PPM (binary P6, RGB)
  {
    std::ofstream f(out_path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    const uint8_t* px = (const uint8_t*)readback.map;
    std::vector<uint8_t> row(W * 3);
    for (uint32_t y = 0; y < H; ++y) {
      for (uint32_t x = 0; x < W; ++x)
        for (int k = 0; k < 3; ++k) row[x * 3 + k] = px[(y * W + x) * 4 + k];
      f.write((const char*)row.data(), (std::streamsize)row.size());
    }
  }
  auto t_end = clk::now();

  // ---- teardown (validation reports leaks, so destroy everything)
  // PROBE_FAULT_LEAK is a negative control: leaking the pipeline must produce validation errors.
  if (!std::getenv("PROBE_FAULT_LEAK")) vkDestroyPipeline(dev, pipeline, nullptr);
  vkDestroyPipelineLayout(dev, layout, nullptr);
  for (auto m : modules) vkDestroyShaderModule(dev, m, nullptr);
  vkDestroyDescriptorPool(dev, dpool, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  vkDestroyImageView(dev, view, nullptr);
  vkDestroyImage(dev, image, nullptr);
  vkFreeMemory(dev, image_mem, nullptr);
  vkDestroyAccelerationStructureKHR(dev, tlas, nullptr);
  vkDestroyAccelerationStructureKHR(dev, blas, nullptr);
  for (Buffer* b : {&vbuf, &ibuf, &nbuf, &blas_buf, &inst_buf, &tlas_buf, &readback, &sbt}) destroy_buffer(c, *b);
  vkDestroyQueryPool(dev, cqpool, nullptr);
  vkDestroyQueryPool(dev, qpool, nullptr);
  vkDestroyCommandPool(dev, pool, nullptr);
  vkDestroyDevice(dev, nullptr);
  if (messenger) {
    auto destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(inst, "vkDestroyDebugUtilsMessengerEXT");
    destroy(inst, messenger, nullptr);
  }
  vkDestroyInstance(inst, nullptr);

  const double mib = 1.0 / (1024.0 * 1024.0);
  std::printf(
      "{\"probe\":\"cpp\",\"scene\":\"%s\",\"camera\":\"%s\",\"sun\":\"%s\",\"sun_dir\":[%.4f,%.4f,%.4f],\"fbx_lights\":%zu,\"device\":\"%s\",\"driver_api\":\"%u.%u.%u\","
      "\"shader_dir\":\"%s\",\"width\":%u,\"height\":%u,\"triangles\":%u,\"vertices\":%u,\"fbx_meshes\":%zu,"
      "\"fbx_nodes\":%zu,\"fbx_cameras\":%zu,\"fbx_textures\":%zu,\"validation\":%s,\"validation_errors\":%d,"
      "\"validation_warnings\":%d,\"geometry_mib\":%.2f,\"blas_mib\":%.2f,\"blas_compacted_mib\":%.2f,"
      "\"blas_scratch_mib\":%.2f,\"tlas_bytes\":%llu,\"vram_before_mib\":%.1f,\"vram_resident_mib\":%.1f,"
      "\"vram_budget_mib\":%.1f,\"host_scene_ms\":%.1f,\"host_startup_to_device_ms\":%.1f,\"host_upload_ms\":%.1f,"
      "\"host_as_total_ms\":%.1f,\"host_pipeline_create_ms\":%.3f,\"host_total_ms\":%.1f,\"gpu_as_build_ms\":%.3f,"
      "\"gpu_trace_cold_ms\":%.3f,\"gpu_trace_warm_median_ms\":%.3f,\"gpu_trace_warm_min_ms\":%.3f,"
      "\"gpu_trace_warm_max_ms\":%.3f}\n",
      scene.name.c_str(), scene.camera_name.c_str(), scene.sun_source.c_str(), scene.cam[16], scene.cam[17], scene.cam[18],
      scene.lights, pr.deviceName, VK_API_VERSION_MAJOR(pr.apiVersion),
      VK_API_VERSION_MINOR(pr.apiVersion), VK_API_VERSION_PATCH(pr.apiVersion), shader_dir.c_str(), W, H, tri_count,
      vert_count, scene.meshes, scene.nodes, scene.cameras, scene.textures, have_validation ? "true" : "false",
      g_validation_errors, g_validation_warnings,
      (double)(scene.pos.size() + scene.idx.size() + scene.nrm.size()) * 4 * mib,
      blas_sz.accelerationStructureSize * mib, blas_compacted * mib, blas_sz.buildScratchSize * mib,
      (unsigned long long)tlas_sz.accelerationStructureSize, vram0 * mib, vram1 * mib, budget1 * mib,
      ms(t_start, t_scene), ms(t_scene, t_device), ms(t_up0, t_up1), ms(t_up0, t_as), ms(t_pipe0, t_pipe1),
      ms(t_start, t_end), (ts[1] - ts[0]) * tick_ms, (ts[3] - ts[2]) * tick_ms, warm[warm.size() / 2], warm.front(),
      warm.back());
  return g_validation_errors ? 3 : 0;
}
