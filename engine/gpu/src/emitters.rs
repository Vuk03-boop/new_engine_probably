//! Phase 4A: the emitter table on the device (ADR-0005 Amendment 3, ADR-0003 Amendment 3).
//!
//! - [`EmitterSet`] keeps the emissive quads of every region with their identity (region key, quad
//!   index in the region, the snapshot the region was built from), so an edit replaces only its
//!   regions' share. `light::emitters::EmitterTable` sorts them by geometry, so the CPU reference and
//!   the GPU sample the same table.
//! - [`GpuEmitter`] is one emitter as the shaders read it (80 bytes); [`RefEmitters`] holds the
//!   emitters and the per-material emitted radiance on the device under `Category::GpuMaterial`.
//! - [`SceneEmitters`] is the table a `GpuScene` publishes with its meshes and TLAS
//!   (`GpuScene::build_lit`); `GpuScene::update` builds the next one before the swap.
//! - 4B part 2: [`changed`] lists the emitters an update changed, for relight (ADR-0006 Amendment 2).

use std::collections::{BTreeMap, BTreeSet};
use std::mem::{offset_of, size_of};

use ash::vk;
use light::emitters::{luminance, Emitter, EmitterId, EmitterQuad, EmitterTable};
use memory::{Category, LedgerError};
use world::{MaterialId, MaterialRegistry};

use crate::alloc::{Allocator, Buffer, Kind};
use crate::context::{Gpu, GpuError, Result};
use crate::layout::{RegionKey, RegionMesh};
use crate::reflect::Field;
use crate::staging::Uploader;
use crate::timeline::Timeline;

/// The emissive quads of every region of a scene, per region.
#[derive(Clone, Debug)]
pub struct EmitterSet {
    registry: MaterialRegistry,
    /// Per material index (as region quads store it): its id, if the material emits.
    emitting: Vec<Option<MaterialId>>,
    quads: BTreeMap<RegionKey, Vec<EmitterQuad>>,
}

impl EmitterSet {
    /// No regions yet. Refuses emitted radiance that is negative or not finite.
    pub fn new(registry: &MaterialRegistry) -> std::result::Result<EmitterSet, String> {
        let emission = light::emitters::emission(registry)?;
        let emitting = registry.iter().zip(&emission).map(|((id, _), l)| (luminance(*l) > 0.0).then_some(id)).collect();
        Ok(EmitterSet { registry: registry.clone(), emitting, quads: BTreeMap::new() })
    }

    /// Region `key` as built from snapshot `snapshot`; `None`: the region has no quads now.
    pub fn set_region(&mut self, key: RegionKey, mesh: Option<&RegionMesh>, snapshot: u64) {
        let quads = mesh.map_or_else(Vec::new, |r| self.emissive_quads(key, r, snapshot));
        if quads.is_empty() {
            self.quads.remove(&key);
        } else {
            self.quads.insert(key, quads);
        }
    }

    fn emissive_quads(&self, key: RegionKey, r: &RegionMesh, snapshot: u64) -> Vec<EmitterQuad> {
        let o = key.origin(r.size);
        r.quads
            .iter()
            .enumerate()
            .filter_map(|(i, q)| {
                // Quads carry registered ids (the mesh is built from the world); an unknown one could
                // only come from a shrunken registry, and emits nothing.
                let material = (*self.emitting.get(q.material as usize)?)?;
                let id = EmitterId { key: [key.x, key.y, key.z], quad: i as u32, snapshot };
                Some(EmitterQuad::from_local(id, material, q.face, q.plane, q.u0, q.v0, q.u1, q.v1, [o.x, o.y, o.z]))
            })
            .collect()
    }

    /// Emissive quads in the set.
    pub fn len(&self) -> usize {
        self.quads.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    /// The table of scene snapshot `snapshot` over every region's emissive quads.
    pub fn table(&self, snapshot: u64) -> std::result::Result<EmitterTable, String> {
        EmitterTable::build(&self.registry, snapshot, self.quads.values().flatten().copied())
    }
}

/// The emitter table of `regions` for scene snapshot `snapshot`, every region at that snapshot.
pub fn table(regions: &BTreeMap<RegionKey, RegionMesh>, registry: &MaterialRegistry, snapshot: u64) -> std::result::Result<EmitterTable, String> {
    let mut set = EmitterSet::new(registry)?;
    for (&k, r) in regions {
        set.set_region(k, Some(r), snapshot);
    }
    set.table(snapshot)
}

/// 4B: the emitters of `regions` whose geometry or radiance is in only one of `old` and `new` (as a
/// multiset; the old one's first, then the new one's, each in table order). A quad split differently
/// counts as changed, which only relights more. Emitters outside `regions` are the same in both.
pub fn changed(old: &EmitterTable, new: &EmitterTable, regions: &BTreeSet<RegionKey>) -> Vec<Emitter> {
    type Key = (u8, [u64; 3], [u64; 3], [u64; 3], [u64; 3]);
    let key = |e: &Emitter| -> Key { (e.face, e.p0.map(f64::to_bits), e.eu.map(f64::to_bits), e.ev.map(f64::to_bits), e.radiance.map(f64::to_bits)) };
    let inside = |e: &&Emitter| regions.contains(&RegionKey { x: e.id.key[0], y: e.id.key[1], z: e.id.key[2] });
    let count = |t: &EmitterTable| {
        let mut m: BTreeMap<Key, usize> = BTreeMap::new();
        for e in t.emitters.iter().filter(inside) {
            *m.entry(key(e)).or_default() += 1;
        }
        m
    };
    let (a, b) = (count(old), count(new));
    let mut out = Vec::new();
    for (t, other) in [(old, &b), (new, &a)] {
        // Of each key, the copies beyond the other table's count.
        let mut seen: BTreeMap<Key, usize> = BTreeMap::new();
        for e in t.emitters.iter().filter(inside) {
            let k = key(e);
            let n = seen.entry(k).or_default();
            *n += 1;
            if *n > other.get(&k).copied().unwrap_or(0) {
                out.push(*e);
            }
        }
    }
    out
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

    /// The rows as the device holds them (little-endian std430).
    pub fn bytes(rows: &[GpuEmitter]) -> Vec<u8> {
        rows.iter().flat_map(|e| e.p0_area.iter().chain(&e.eu_pdf).chain(&e.ev).chain(&e.radiance).flat_map(|x| x.to_le_bytes()).chain(e.meta.iter().flat_map(|x| x.to_le_bytes())).collect::<Vec<u8>>()).collect()
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

/// The per-material emitted radiance as the device holds it (RGBA32F, little-endian).
pub fn emission_bytes(table: &EmitterTable) -> Vec<u8> {
    table.emission.iter().flat_map(|l| [l[0] as f32, l[1] as f32, l[2] as f32, 0.0]).flat_map(f32::to_le_bytes).collect()
}

impl RefEmitters {
    /// Uploads `table`; returns the timeline value after which it is on the device.
    pub fn upload(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, table: &EmitterTable) -> Result<(RefEmitters, u64)> {
        let rows = GpuEmitter::of(table);
        let bytes = GpuEmitter::bytes(&rows);
        let em = emission_bytes(table);
        // TRANSFER_SRC: the checks read the table back (4A G6).
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC;
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

    /// Device bytes of both buffers.
    pub fn device_bytes(&self) -> u64 {
        self.emitters.range().2 + self.emission.range().2
    }

    pub fn free(self, gpu: &Gpu, alloc: &mut Allocator) {
        alloc.free(gpu, self.emitters);
        alloc.free(gpu, self.emission);
    }
}

/// The emitter table a `GpuScene` publishes with its meshes and TLAS (ADR-0003 Amendment 3): the
/// regions' emissive quads, the table of the snapshot shown, and that table on the device.
pub struct SceneEmitters {
    pub set: EmitterSet,
    pub table: EmitterTable,
    pub device: RefEmitters,
}

impl SceneEmitters {
    /// The table of `set` for snapshot `snapshot`, uploaded. `refuse` plants a refused grant (as by the
    /// budget) before anything is allocated. Returns the timeline value of the upload.
    pub fn publish(gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, set: EmitterSet, snapshot: u64, refuse: bool) -> Result<(SceneEmitters, u64)> {
        let table = set.table(snapshot).map_err(|e| GpuError::Layout(vec![format!("emitter table: {e}")]))?;
        if refuse {
            let requested = (table.len() * size_of::<GpuEmitter>()) as u64;
            return Err(GpuError::OverBudget(LedgerError::OverBudget { category: Category::GpuMaterial, requested, category_reserved: 0, total_reserved: 0, limit: 0, limit_is_total: false }));
        }
        let (device, v) = RefEmitters::upload(gpu, alloc, up, timeline, &table)?;
        Ok((SceneEmitters { set, table, device }, v))
    }

    /// The next table: `changes` (region, its mesh at `snapshot` or `None`) applied to a copy of the
    /// set, then [`SceneEmitters::publish`]. `self` is unchanged, so a refusal keeps it whole.
    #[allow(clippy::too_many_arguments)]
    pub fn next<'a>(&self, gpu: &Gpu, alloc: &mut Allocator, up: &mut Uploader, timeline: &mut Timeline, changes: impl IntoIterator<Item = (RegionKey, Option<&'a RegionMesh>)>, snapshot: u64, refuse: bool) -> Result<(SceneEmitters, u64)> {
        let mut set = self.set.clone();
        for (k, m) in changes {
            set.set_region(k, m, snapshot);
        }
        Self::publish(gpu, alloc, up, timeline, set, snapshot, refuse)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::layout::{build_regions, RegionSize};
    use crate::scene::{affected_regions, region_meshes};
    use derived::{extract_world, Config, Merge, Pipeline};
    use world::dims::BRICK_EDGE;
    use world::scene::{street_night, Dressing};
    use world::{BrickKey, Transaction, VoxelCoord, World};

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

    /// 4B: `changed` lists the emitters in only one of two tables, within the rebuilt regions, as a
    /// multiset: moved, removed and recoloured quads count; an unchanged region does not.
    #[test]
    fn changed_lists_what_differs() {
        let mut r = MaterialRegistry::new();
        let hot = r.register("hot", world::MaterialParams { base_color: [0.5; 3], emissive: [10.0, 5.0, 1.0] }).unwrap();
        let warm = r.register("warm", world::MaterialParams { base_color: [0.5; 3], emissive: [1.0, 1.0, 1.0] }).unwrap();
        let q = |region: i32, i: u32, m, u0: i32| EmitterQuad { id: EmitterId { key: [region, 0, 0], quad: i, snapshot: 1 }, material: m, face: 3, plane: 0, u0, v0: 0, u1: u0 + 1, v1: 1 };
        let old = EmitterTable::build(&r, 1, [q(0, 0, hot, 0), q(0, 1, hot, 2), q(0, 2, hot, 4), q(0, 3, hot, 6), q(1, 0, hot, 40)]).unwrap();
        // Region 0: quad 0 kept (new index), quad 1 moved, quad 2 removed, quad 3 recoloured. Region 1
        // moved too, but it is not in the rebuilt set.
        let new = EmitterTable::build(&r, 2, [q(0, 5, hot, 0), q(0, 6, hot, 3), q(0, 7, warm, 6), q(1, 0, hot, 41)]).unwrap();
        let regions: BTreeSet<RegionKey> = [RegionKey { x: 0, y: 0, z: 0 }].into();
        // Face 3 lies on y; u runs along z.
        let got: Vec<(f64, f64)> = changed(&old, &new, &regions).iter().map(|e| (e.p0[2], e.radiance[1])).collect();
        let set = |v: &[(f64, f64)]| v.iter().map(|&(u, g)| (u as i32, g as i32)).collect::<BTreeSet<_>>();
        assert_eq!(set(&got), set(&[(2.0, 5.0), (4.0, 5.0), (6.0, 5.0), (3.0, 5.0), (6.0, 1.0)]));
        assert_eq!(got.len(), 5, "{got:?}: old 2, 4, 6 (hot), new 3 (hot) and 6 (warm)");
        let all: BTreeSet<RegionKey> = [RegionKey { x: 0, y: 0, z: 0 }, RegionKey { x: 1, y: 0, z: 0 }].into();
        assert_eq!(changed(&old, &new, &all).len(), 7);
        assert!(changed(&old, &old, &all).is_empty());
    }

    /// Runs every job inline and publishes; the published brick keys.
    fn drain(p: &mut Pipeline, w: &World) -> BTreeSet<BrickKey> {
        loop {
            let jobs = p.dispatch(w);
            if jobs.is_empty() {
                break;
            }
            for j in jobs {
                p.complete(w, j.run());
            }
        }
        let published = p.try_publish(w);
        assert!(p.is_idle(), "pipeline did not drain");
        published.map(|x| x.groups.into_iter().flatten().collect()).unwrap_or_default()
    }

    /// Every region of the pipeline's current snapshot, built from scratch.
    fn full(p: &mut Pipeline, size: RegionSize) -> BTreeMap<RegionKey, RegionMesh> {
        let t = p.acquire();
        let keys: Vec<BrickKey> = p.current().keys().collect();
        let meshes: Vec<_> = keys.iter().map(|&k| (k, p.read(&t, k).unwrap().unwrap().clone())).collect();
        p.release(t).unwrap();
        build_regions(meshes.iter().map(|(k, m)| (*k, m)), size)
    }

    /// One emitter without the snapshot of its id: geometry, radiance, sampling, region and quad.
    type Row = (u8, [f64; 3], [f64; 3], [f64; 3], [f64; 3], f64, u32, u32, [i32; 3], u32);

    /// Voxel edits: (voxel, new material).
    type Edits = Vec<(VoxelCoord, Option<world::MaterialId>)>;

    /// Everything but the snapshot in the ids, in table order.
    fn content(t: &EmitterTable) -> Vec<Row> {
        t.emitters.iter().map(|e| (e.face, e.p0, e.eu, e.ev, e.radiance, e.pdf, e.threshold, e.alias, e.id.key, e.id.quad)).collect()
    }

    /// C6 (4A part 2): the scene's incremental table equals a from-scratch build after every edit of
    /// the sequence, through the real 1C pipeline, with ids that follow the edited regions.
    #[test]
    fn edits_keep_the_table_equal_to_a_fresh_build() {
        let size = RegionSize::Chunk;
        let (mut w, _) = street_night(Dressing::Full);
        let mut p = Pipeline::new(Config { merge: Merge::Greedy, ..Config::default() });
        p.mark_all(&w);
        drain(&mut p, &w);
        let snap0 = p.current().id.raw();
        let mut set = EmitterSet::new(w.materials()).unwrap();
        for (k, m) in &full(&mut p, size) {
            set.set_region(*k, Some(m), snap0);
        }
        let original = set.table(snap0).unwrap();
        assert_eq!(content(&original), content(&table(&full(&mut p, size), w.materials(), snap0).unwrap()));
        assert!(original.len() > 1000);

        let mats = w.materials().clone();
        let id = |n: &str| mats.id_of(n).unwrap();
        let first = |w: &World, m| w.occupied().find(|&(_, x)| x == m).map(|(v, _)| v).unwrap();
        let lamp = first(&w, id("lamp"));
        let bulb = first(&w, id("bulb"));
        let air = VoxelCoord::new(136, 40, 64);
        assert_eq!(w.get(air), None, "open air above the road");
        let region_of = |v: VoxelCoord| RegionKey::of(v.split().0, size);
        let emitting = |s: &EmitterSet, r: RegionKey| s.quads.contains_key(&r);
        // A non-emitting voxel inside its brick, in a region without emitters: its edit touches no light.
        let inside = |c: i32| (1..BRICK_EDGE - 1).contains(&c.rem_euclid(BRICK_EDGE));
        let (quiet, quiet_mat) = w.occupied().find(|&(v, m)| set.emitting[m.raw() as usize].is_none() && inside(v.x) && inside(v.y) && inside(v.z) && !emitting(&set, region_of(v))).unwrap();

        let steps: Vec<(&str, Edits)> = vec![
            ("remove a lamp voxel", vec![(lamp, None)]),
            ("remove a bulb voxel", vec![(bulb, None)]),
            ("add an emissive voxel in open air", vec![(air, Some(id("neon_pink")))]),
            ("remove a voxel in a region without emitters", vec![(quiet, None)]),
            ("restore everything", vec![(lamp, Some(id("lamp"))), (bulb, Some(id("bulb"))), (air, None), (quiet, Some(quiet_mat))]),
        ];
        let mut prev = original.clone();
        let mut edited_ever = BTreeSet::new();
        for (label, edits) in steps {
            let mut tx = Transaction::new();
            for &(v, m) in &edits {
                tx.set(v, m);
            }
            let applied = w.apply(&tx).unwrap();
            assert!(!applied.changed.is_empty(), "{label}: the edit changes the world");
            p.notify_edits(&applied.changed);
            let keys = drain(&mut p, &w);
            let snap = p.current().id.raw();
            let regions = affected_regions(keys, size);
            let token = p.acquire();
            let changed = region_meshes(&p, &token, &regions, size).unwrap();
            p.release(token).unwrap();
            let before = set.clone();
            for (k, m) in &changed {
                set.set_region(*k, m.as_ref(), snap);
            }
            let inc = set.table(snap).unwrap();
            let scratch = table(&full(&mut p, size), w.materials(), snap).unwrap();
            assert_eq!(content(&inc), content(&scratch), "{label}: incremental and from-scratch tables differ");
            assert_eq!(inc.snapshot, snap);
            let old: BTreeMap<_, _> = prev.emitters.iter().map(|e| ((e.id.key, e.id.quad), e.id.snapshot)).collect();
            for e in &inc.emitters {
                let r = RegionKey { x: e.id.key[0], y: e.id.key[1], z: e.id.key[2] };
                let want = if regions.contains(&r) { snap } else { old[&(e.id.key, e.id.quad)] };
                assert_eq!(e.id.snapshot, want, "{label}: emitter {:?}", e.id);
            }
            eprintln!("C6 {label}: {} regions rebuilt, {} emitters (was {}), snapshot {snap}", regions.len(), inc.len(), prev.len());
            match label {
                "remove a lamp voxel" => {
                    assert!(inc.len() != prev.len() || content(&inc) != content(&prev), "the lamp edit changes the table");
                    // Negative control: the lamp's region left out of the update is caught.
                    let mut faulty = before.clone();
                    for (k, m) in changed.iter().filter(|(k, _)| **k != region_of(lamp)) {
                        faulty.set_region(*k, m.as_ref(), snap);
                    }
                    assert_ne!(content(&faulty.table(snap).unwrap()), content(&scratch), "a missed region must be caught");
                }
                "remove a voxel in a region without emitters" => {
                    assert!(regions.iter().all(|&r| !emitting(&before, r) && !emitting(&set, r)), "the quiet edit's regions hold no emitters");
                    assert_eq!(inc.emitters.iter().map(|e| e.id).collect::<Vec<_>>(), prev.emitters.iter().map(|e| e.id).collect::<Vec<_>>(), "no emitter id changes");
                }
                "restore everything" => {
                    let geometry = |t: &EmitterTable| t.emitters.iter().map(|e| (e.face, e.p0, e.eu, e.ev, e.radiance, e.pdf, e.threshold, e.alias)).collect::<Vec<_>>();
                    assert_eq!(geometry(&inc), geometry(&original), "restoring gives the original table back");
                    edited_ever.extend(regions.iter().copied());
                    for (a, o) in inc.emitters.iter().zip(&original.emitters) {
                        let r = RegionKey { x: a.id.key[0], y: a.id.key[1], z: a.id.key[2] };
                        assert_eq!(a.id.snapshot == o.id.snapshot, !edited_ever.contains(&r), "{:?}", a.id);
                    }
                }
                _ => {}
            }
            edited_ever.extend(regions.iter().copied());
            prev = inc;
        }
    }
}
