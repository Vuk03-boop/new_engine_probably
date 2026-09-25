//! Phase 4A: the emitter table on the device (ADR-0005 Amendment 3, ADR-0003 Amendment 3).
//!
//! - [`region_quads`] turns region meshes into `light::emitters::EmitterQuad`s with their identity
//!   (region key, quad index in the region, the region's snapshot). `light::emitters::EmitterTable`
//!   sorts them by geometry, so the CPU reference and the GPU sample the same table.
//! - [`GpuEmitter`] is one emitter as the shaders read it (80 bytes); [`RefEmitters`] holds the
//!   emitters and the per-material emitted radiance on the device under `Category::GpuMaterial`.

use std::collections::BTreeMap;
use std::mem::{offset_of, size_of};

use ash::vk;
use light::emitters::{EmitterId, EmitterQuad, EmitterTable};
use memory::Category;
use world::MaterialRegistry;

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, Result};
use crate::layout::{RegionKey, RegionMesh};
use crate::reflect::Field;
use crate::staging::Uploader;
use crate::timeline::Timeline;

/// Every quad of `regions` as an emitter candidate; `snapshot` gives each region's snapshot.
pub fn region_quads<'a>(regions: impl IntoIterator<Item = (&'a RegionKey, &'a RegionMesh)>, registry: &MaterialRegistry, snapshot: impl Fn(RegionKey) -> u64) -> Vec<EmitterQuad> {
    let ids: Vec<_> = registry.iter().map(|(id, _)| id).collect();
    let mut out = Vec::new();
    for (&k, r) in regions {
        let o = k.origin(r.size);
        let s = snapshot(k);
        for (i, q) in r.quads.iter().enumerate() {
            // Quads carry registered ids (the mesh is built from the world); an unknown one is a bug
            // that `EmitterTable::build` would refuse, so it is skipped only if the registry shrank.
            let Some(&material) = ids.get(q.material as usize) else {
                continue;
            };
            let id = EmitterId { key: [k.x, k.y, k.z], quad: i as u32, snapshot: s };
            out.push(EmitterQuad::from_local(id, material, q.face, q.plane, q.u0, q.v0, q.u1, q.v1, [o.x, o.y, o.z]));
        }
    }
    out
}

/// The emitter table of `regions` for scene snapshot `snapshot`, every region at that snapshot.
pub fn table(regions: &BTreeMap<RegionKey, RegionMesh>, registry: &MaterialRegistry, snapshot: u64) -> std::result::Result<EmitterTable, String> {
    EmitterTable::build(registry, snapshot, region_quads(regions, registry, |_| snapshot))
}

/// One emitter as `reference.slang` reads it (std430, 80 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuEmitter {
    /// xyz: corner (u0, v0), voxels; w: area.
    pub p0_area: [f32; 4],
    /// xyz: edge along u; w: realized selection probability.
    pub eu_pdf: [f32; 4],
    /// xyz: edge along v; w: unused.
    pub ev: [f32; 4],
    /// rgb: emitted radiance; w: unused.
    pub radiance: [f32; 4],
    /// x: alias threshold (0..=2²⁴), y: alias index, z: face (ADR-0004), w: unused.
    pub meta: [u32; 4],
}

const _: () = assert!(size_of::<GpuEmitter>() == 80, "ADR-0003 Amendment 3: 80 bytes per emitter");

impl GpuEmitter {
    pub fn of(table: &EmitterTable) -> Vec<GpuEmitter> {
        let f = |v: [f64; 3], w: f64| [v[0] as f32, v[1] as f32, v[2] as f32, w as f32];
        table
            .emitters
            .iter()
            .map(|e| GpuEmitter { p0_area: f(e.p0, e.area), eu_pdf: f(e.eu, e.pdf), ev: f(e.ev, 0.0), radiance: f(e.radiance, 0.0), meta: [e.threshold, e.alias, e.face as u32, 0] })
            .collect()
    }

    /// The fields the host assumes (checked against the shader's reflection).
    pub fn fields() -> Vec<Field> {
        let d = GpuEmitter::default();
        let f = |name: &'static str, offset: usize, size: usize| Field { name, offset: offset as u64, size: size as u64 };
        vec![
            f("p0_area", offset_of!(GpuEmitter, p0_area), size_of_val(&d.p0_area)),
            f("eu_pdf", offset_of!(GpuEmitter, eu_pdf), size_of_val(&d.eu_pdf)),
            f("ev", offset_of!(GpuEmitter, ev), size_of_val(&d.ev)),
            f("radiance", offset_of!(GpuEmitter, radiance), size_of_val(&d.radiance)),
            f("meta", offset_of!(GpuEmitter, meta), size_of_val(&d.meta)),
        ]
    }
}

/// The emitters and the per-material emitted radiance (RGBA32F) of one table, on the device.
pub struct RefEmitters {
    pub emitters: Buffer,
    pub emission: Buffer,
    pub count: u32,
    pub materials: u32,
    pub snapshot: u64,
}

impl RefEmitters {
    /// Uploads `table`; returns the timeline value after which it is on the device.
    pub fn upload(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, table: &EmitterTable) -> Result<(RefEmitters, u64)> {
        let rows = GpuEmitter::of(table);
        let bytes: Vec<u8> = rows.iter().flat_map(|e| e.p0_area.iter().chain(&e.eu_pdf).chain(&e.ev).chain(&e.radiance).flat_map(|x| x.to_le_bytes()).chain(e.meta.iter().flat_map(|x| x.to_le_bytes())).collect::<Vec<u8>>()).collect();
        let em: Vec<u8> = table.emission.iter().flat_map(|l| [l[0] as f32, l[1] as f32, l[2] as f32, 0.0]).flat_map(f32::to_le_bytes).collect();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let emitters = alloc.create_buffer(gpu, (bytes.len() as u64).max(size_of::<GpuEmitter>() as u64), usage, Category::GpuMaterial, Kind::Device)?;
        let emission = match alloc.create_buffer(gpu, (em.len() as u64).max(16), usage, Category::GpuMaterial, Kind::Device) {
            Ok(b) => b,
            Err(e) => {
                alloc.free(gpu, emitters);
                return Err(e);
            }
        };
        let v = (|| {
            if !bytes.is_empty() {
                up.upload(gpu, timeline, &emitters, 0, &bytes)?;
            }
            if !em.is_empty() {
                up.upload(gpu, timeline, &emission, 0, &em)?;
            }
            up.flush(gpu, timeline)
        })();
        match v {
            Ok(v) => Ok((RefEmitters { emitters, emission, count: rows.len() as u32, materials: table.emission.len() as u32, snapshot: table.snapshot }, v)),
            Err(e) => {
                alloc.free(gpu, emitters);
                alloc.free(gpu, emission);
                Err(e)
            }
        }
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free(gpu, self.emitters);
        alloc.free(gpu, self.emission);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{build_regions, RegionSize};
    use derived::{extract_world, Merge};
    use world::scene::{street_night, Dressing};

    /// C4 on the GPU's own grouping: every region size gives the same table (order, geometry,
    /// probabilities), with region identities.
    #[test]
    fn region_size_does_not_change_the_table() {
        let (w, _) = street_night(Dressing::Full);
        let meshes: Vec<_> = w.bricks().map(|(k, _)| (k, extract_world(&w, k, Merge::Greedy).unwrap())).collect();
        let tables: Vec<EmitterTable> = [RegionSize::Brick, RegionSize::Chunk, RegionSize::Chunks2]
            .iter()
            .map(|&size| table(&build_regions(meshes.iter().map(|(k, m)| (*k, m)), size), w.materials(), 7).unwrap())
            .collect();
        assert!(tables[0].len() > 1000);
        for t in &tables[1..] {
            let g = |t: &EmitterTable| t.emitters.iter().map(|e| (e.face, e.p0, e.eu, e.ev, e.pdf, e.threshold, e.alias)).collect::<Vec<_>>();
            assert_eq!(g(t), g(&tables[0]));
        }
        assert!(tables[1].emitters.iter().all(|e| e.id.snapshot == 7));
        let rows = GpuEmitter::of(&tables[1]);
        assert_eq!(rows.len(), tables[1].len());
        assert!(rows.iter().all(|r| r.meta[0] <= light::emitters::COIN_ONE && (r.meta[1] as usize) < rows.len() && r.meta[2] < 6));
    }
}
