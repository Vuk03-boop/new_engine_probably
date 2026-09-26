//! Compiles `shaders/*.slang` with the pinned `slangc` (ADR-0001: Vulkan SDK 1.4.357.0, slangc
//! 2026.13.1), validates the SPIR-V with `spirv-val`, and writes each module's reflection JSON next
//! to it in `OUT_DIR`. A different compiler version is a build error, not a warning.

use std::path::PathBuf;
use std::process::Command;

const SLANGC_VERSION: &str = "2026.13.1";
/// (source file, entry point, stage, output stem, extra slangc arguments)
const SHADERS: &[(&str, &str, &str, &str, &[&str])] = &[
    ("mesh_decode.slang", "main", "compute", "mesh_decode", &[]),
    ("raster.slang", "vs_main", "vertex", "raster_vs", &[]),
    ("raster.slang", "fs_main", "fragment", "raster_fs", &[]),
    ("debug_view.slang", "vs_main", "vertex", "debug_view_vs", &[]),
    ("debug_view.slang", "fs_main", "fragment", "debug_view_fs", &[]),
    // 2D: ray query is declared up front, so slangc does not warn about upgrading the profile.
    ("ray_primary.slang", "main", "compute", "ray_primary", &["-capability", "spvRayQueryKHR"]),
    // 3A: the reference path tracer (ADR-0005).
    ("reference.slang", "main", "compute", "reference", &["-capability", "spvRayQueryKHR"]),
    // 3B: real-time lighting from the G-buffer.
    ("shade.slang", "main", "compute", "shade", &["-capability", "spvRayQueryKHR"]),
    // 4B: the same file with the emitter terms (the lit module). Without EMITTERS it is the M3 one.
    ("shade.slang", "main", "compute", "shade_lit", &["-capability", "spvRayQueryKHR", "-D", "EMITTERS"]),
    // 3C: the sky-view table.
    ("sky.slang", "main", "compute", "sky", &[]),
    // S-020: the sky correction's bake (a tool, not a frame pass).
    ("sky_bake.slang", "main", "compute", "sky_bake", &[]),
    // 3D: the temporal foundation.
    ("temporal.slang", "main", "compute", "temporal", &[]),
    // 3E: the native reconstruction.
    ("denoise.slang", "main", "compute", "denoise", &[]),
    // 4B: emission after reconstruction and the exposure meter.
    ("compose.slang", "main", "compute", "compose", &[]),
];

fn main() {
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-changed=build.rs");
    // Shared includes (light_common.slang, sky_common.slang, emitters_common.slang) are not listed as modules.
    println!("cargo:rerun-if-changed=shaders/light_common.slang");
    println!("cargo:rerun-if-changed=shaders/sky_common.slang");
    println!("cargo:rerun-if-changed=shaders/emitters_common.slang");
    let sdk = std::env::var("VULKAN_SDK").expect("VULKAN_SDK is not set: the gpu crate needs the pinned Vulkan SDK (ADR-0001)");
    let bin = PathBuf::from(sdk).join("Bin");
    let slangc = bin.join("slangc.exe");
    let spirv_val = bin.join("spirv-val.exe");
    let out = Command::new(&slangc).arg("-v").output().unwrap_or_else(|e| panic!("cannot run {}: {e}", slangc.display()));
    let version = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(version.trim().starts_with(SLANGC_VERSION), "slangc {} is not the pinned {SLANGC_VERSION}", version.trim());

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for &(file, entry, stage, stem, extra) in SHADERS {
        let src = PathBuf::from("shaders").join(file);
        println!("cargo:rerun-if-changed={}", src.display());
        let spv = out_dir.join(format!("{stem}.spv"));
        let json = out_dir.join(format!("{stem}.json"));
        let status = Command::new(&slangc)
            .arg(&src)
            .args(["-target", "spirv", "-profile", "spirv_1_5"])
            .args(extra)
            .args(["-entry", entry, "-stage", stage, "-o"])
            .arg(&spv)
            .arg("-reflection-json")
            .arg(&json)
            .status()
            .expect("run slangc");
        assert!(status.success(), "slangc failed on {file}");
        let status = Command::new(&spirv_val).args(["--target-env", "vulkan1.3"]).arg(&spv).status().expect("run spirv-val");
        assert!(status.success(), "spirv-val rejected {file}");
        let caps = capabilities(&std::fs::read(&spv).expect("read SPIR-V"));
        let extra: Vec<u32> = caps.iter().copied().filter(|c| !ALLOWED_CAPABILITIES.contains(c)).collect();
        assert!(extra.is_empty(), "{file} ({entry}) declares SPIR-V capabilities {extra:?} beyond {ALLOWED_CAPABILITIES:?}; each one needs a device feature enabled in gpu::context first");
    }
}

/// SPIR-V capabilities the device creation in `src/context.rs` supports: 1 = Shader, 2 = Geometry
/// (fragment `SV_PrimitiveID`; `geometryShader` is enabled and required), 4472 = RayQueryKHR
/// (`rayQuery`, required), 5347 = PhysicalStorageBufferAddresses (`bufferDeviceAddress`, enabled).
/// A new one (e.g. 22 = Int16, 9 = Float16, 11 = Int64) must first be enabled as a device feature,
/// then added here.
const ALLOWED_CAPABILITIES: &[u32] = &[1, 2, 4472, 5347];

/// The operands of every `OpCapability` (opcode 17) in a SPIR-V module.
fn capabilities(spv: &[u8]) -> Vec<u32> {
    let words: Vec<u32> = spv.as_chunks::<4>().0.iter().map(|&c| u32::from_le_bytes(c)).collect();
    assert_eq!(words.first(), Some(&0x0723_0203), "SPIR-V magic");
    let mut out = Vec::new();
    let mut i = 5; // header words
    while i < words.len() {
        let (count, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(count > 0, "malformed SPIR-V at word {i}");
        if opcode == 17 {
            out.push(words[i + 1]);
        }
        i += count;
    }
    out
}
