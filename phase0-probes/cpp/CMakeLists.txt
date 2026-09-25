cmake_minimum_required(VERSION 3.20)
project(rt_probe_cpp C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Vulkan REQUIRED)

# ufbx v0.23.0, identical to the copy bundled in the Rust `ufbx` 0.11.4 crate (see ../third_party/ufbx).
add_library(ufbx STATIC ../third_party/ufbx/ufbx.c)
target_include_directories(ufbx PUBLIC ../third_party/ufbx)

add_executable(rt_probe main.cpp)
target_link_libraries(rt_probe PRIVATE Vulkan::Vulkan ufbx)
if(MSVC)
  target_compile_options(rt_probe PRIVATE /W4 /permissive-)
  target_compile_definitions(rt_probe PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
