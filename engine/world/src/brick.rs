//! Brick payload: an occupancy mask plus a material-ID array, kept as separate fields.

use crate::coords::VoxelIndex;
use crate::dims::{BRICK_VOXELS, OCCUPANCY_WORDS};
use crate::material::MaterialId;

/// World-wide content version stamp. Every content change takes the next value from one monotonic
/// counter, so a brick that is removed and recreated never reuses an earlier version.
/// `ContentVersion(0)` means "nothing was ever written" and is never assigned to a brick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentVersion(pub(crate) u64);

impl ContentVersion {
    pub const NONE: ContentVersion = ContentVersion(0);

    pub fn raw(self) -> u64 {
        self.0
    }
}

/// Always non-empty while stored in a world; the world frees a brick when its last voxel is cleared.
///
/// Canonical form: the material slot of an unoccupied voxel always holds the ID with raw value 0.
/// Two bricks with the same occupancy and the same materials on occupied voxels are therefore
/// byte-identical, and [`Brick::same_content`] can compare them directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Brick {
    occupancy: [u64; OCCUPANCY_WORDS],
    materials: Box<[MaterialId; BRICK_VOXELS]>,
    occupied: u16,
    version: ContentVersion,
}

const CANONICAL_EMPTY: MaterialId = MaterialId::from_raw(0);

impl Brick {
    pub(crate) fn new(version: ContentVersion) -> Self {
        Self { occupancy: [0; OCCUPANCY_WORDS], materials: Box::new([CANONICAL_EMPTY; BRICK_VOXELS]), occupied: 0, version }
    }

    /// Rebuilds a stored brick from saved parts (Phase 1B). Rejects anything the world API could
    /// not have produced: an empty brick, a non-canonical unoccupied slot, or the `NONE` version.
    pub(crate) fn from_parts(occupancy: [u64; OCCUPANCY_WORDS], materials: Box<[MaterialId; BRICK_VOXELS]>, version: ContentVersion) -> Result<Self, &'static str> {
        let occupied = occupancy.iter().map(|w| w.count_ones()).sum::<u32>();
        if occupied == 0 {
            return Err("empty brick");
        }
        if version == ContentVersion::NONE {
            return Err("brick without a content version");
        }
        let unoccupied_canonical = (0..BRICK_VOXELS).all(|i| occupancy[i / 64] >> (i % 64) & 1 == 1 || materials[i] == CANONICAL_EMPTY);
        if !unoccupied_canonical {
            return Err("non-canonical material in an unoccupied slot");
        }
        Ok(Self { occupancy, materials, occupied: occupied as u16, version })
    }

    pub fn get(&self, v: VoxelIndex) -> Option<MaterialId> {
        let i = v.get();
        if self.occupancy[i / 64] >> (i % 64) & 1 == 1 {
            Some(self.materials[i])
        } else {
            None
        }
    }

    /// Returns true if the content changed. The caller stamps the version.
    pub(crate) fn put(&mut self, v: VoxelIndex, m: Option<MaterialId>) -> bool {
        let i = v.get();
        let bit = 1u64 << (i % 64);
        let was = self.occupancy[i / 64] & bit != 0;
        match m {
            Some(id) => {
                if was && self.materials[i] == id {
                    return false;
                }
                if !was {
                    self.occupancy[i / 64] |= bit;
                    self.occupied += 1;
                }
                self.materials[i] = id;
            }
            None => {
                if !was {
                    return false;
                }
                self.occupancy[i / 64] &= !bit;
                self.occupied -= 1;
                self.materials[i] = CANONICAL_EMPTY;
            }
        }
        true
    }

    pub(crate) fn stamp(&mut self, version: ContentVersion) {
        self.version = version;
    }

    /// Number of occupied voxels, `1..=BRICK_VOXELS` while stored in a world.
    pub fn occupied_count(&self) -> usize {
        self.occupied as usize
    }

    pub fn is_empty(&self) -> bool {
        self.occupied == 0
    }

    pub fn version(&self) -> ContentVersion {
        self.version
    }

    pub fn occupancy_words(&self) -> &[u64; OCCUPANCY_WORDS] {
        &self.occupancy
    }

    /// Compares occupancy and materials, ignoring the version stamp.
    pub fn same_content(&self, other: &Brick) -> bool {
        self.occupancy == other.occupancy && self.materials == other.materials
    }

    /// Occupied voxels and their materials, in slot order.
    pub fn voxels(&self) -> impl Iterator<Item = (VoxelIndex, MaterialId)> + '_ {
        VoxelIndex::all().filter_map(move |v| self.get(v).map(|m| (v, m)))
    }
}
