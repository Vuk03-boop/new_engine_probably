//! Transactional derived data (PROPOSITION §2): version-tagged jobs, a bounded queue with a
//! deterministic overflow fallback, atomic publication of edit sets and reader-safe retirement.
//!
//! # Contract
//!
//! 1. **Dirty tracking.** `notify_edits` is one edit batch. It marks every brick whose result can
//!    change (the edited brick, plus face neighbours for boundary voxels) as *pending*, and puts them
//!    in one **publication group**. A batch that touches a key of an unpublished group merges with that
//!    group. Pending keys are always eventually queued again.
//! 2. **Jobs** capture immutable copies of their seven input bricks. They run anywhere (`Job::run` is
//!    pure) and can finish in any order.
//! 3. **Rejection.** A result is rejected if its job was *cancelled* (its target was dirtied again
//!    after dispatch), or if it is *stale*: an input's relevant content (the target brick, or a
//!    neighbour's face layer) differs from the world now. Versions are the fast path; content decides
//!    on a mismatch. Cancellation and the stale check are independent guards. A rejected key stays
//!    pending and is queued again.
//! 4. **Publication.** Accepted results wait in *staging* inside their group. `try_publish`
//!    re-validates staging, then publishes, in one new snapshot, every group whose members are all
//!    staged. Revalidation skips results already checked at the current world version, and only
//!    groups that gained a staged member since the last call are checked for readiness, so a call
//!    costs O(changed work), not O(all staged work). Keys of unfinished groups keep their previous value, so each published edit set appears
//!    all at once (for example, both sides of a boundary edit), never half-applied.
//! 5. **Retirement.** A replaced or removed result is freed only after every reader that could see it
//!    (holding a snapshot at or before the last one containing it) has released. Freed slots bump
//!    their generation, so a reused slot never answers an old handle.
//! 6. **Overload.** The queue holds at most `queue_capacity` keys, and dispatch keeps at most
//!    `max_in_flight` jobs. Keys that do not fit go to an ordered overflow set, which refills the
//!    queue in key order. Nothing is dropped, and the order is deterministic.
//!
//! **Sharing (2E).** A snapshot's entries are sharded by chunk, and shards are shared between
//! snapshots (`Arc`). A publication copies the chunk index (one pointer per chunk) and only the
//! shards it changes, so its cost is O(chunks + changed shards), not O(entries). 1C cloned the whole
//! table; the 1D observation of a 4.75 MB publish transient came from that.
//! A region edited faster than it can be re-derived never publishes (its group keeps merging);
//! `Stats::publish_deferred` and `max_group_len` expose this.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use world::{BrickKey, ChunkCoord, ContentVersion, VoxelCoord, World};

use crate::slots::{Handle, PoolStats, SlotPool};
use crate::mesh::{BrickMesh, Merge};
use crate::surface::{affected_keys, Dependencies, InputRecord, InputSnapshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotId(u64);

impl SnapshotId {
    /// The snapshot's number (publication order), for display and logs.
    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReaderId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GroupId(u64);

/// Planted faults for negative-control tests. Each disables exactly one guard; all are off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Faults {
    /// Boundary edits mark only their own brick, not face neighbours.
    pub skip_neighbour_dirty: bool,
    /// Re-dirtying a key does not cancel its in-flight job.
    pub skip_cancel: bool,
    /// Completion and publication do not validate dependencies.
    pub skip_stale_check: bool,
    /// Replaced results are freed at publication, ignoring readers.
    pub retire_without_readers: bool,
    /// Every accepted result publishes immediately, without waiting for the rest of its group.
    pub publish_partial_groups: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub queue_capacity: usize,
    pub max_in_flight: usize,
    /// Mesh merge extent (ADR-0003 granularity parameter). Every job of a pipeline uses the same one.
    pub merge: Merge,
    pub faults: Faults,
}

impl Default for Config {
    fn default() -> Self {
        Self { queue_capacity: 256, max_in_flight: 64, merge: Merge::default(), faults: Faults::default() }
    }
}

/// A unit of work. Send it anywhere, call [`Job::run`], and hand the result back to `complete`.
#[derive(Debug)]
pub struct Job {
    pub id: JobId,
    pub key: BrickKey,
    pub merge: Merge,
    pub inputs: InputSnapshot,
}

#[derive(Clone, Debug)]
pub struct JobResult {
    pub id: JobId,
    pub key: BrickKey,
    pub deps: Dependencies,
    pub output: Option<BrickMesh>,
}

impl Job {
    /// Heap held by the job's input copies (Phase 1D). Jobs live with their caller, so this is not
    /// part of [`Pipeline::memory`].
    pub fn heap(&self) -> memory::Usage {
        self.inputs.heap()
    }

    pub fn run(self) -> JobResult {
        let output = self.inputs.extract(self.merge);
        JobResult { id: self.id, key: self.key, deps: self.inputs.dependencies(), output }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// The target was dirtied again after this job was dispatched.
    Cancelled,
    /// A dependency changed since dispatch; `changed` is the first such brick.
    Stale { changed: BrickKey },
    /// Not an in-flight job of this pipeline (for example, a duplicate completion).
    UnknownJob,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    Accepted,
    Rejected(Rejection),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadError {
    UnknownReader,
    /// The snapshot's resource no longer resolves: a retirement bug. Must never happen.
    Dangling { key: BrickKey },
}

/// What one successful `try_publish` changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Published {
    pub id: SnapshotId,
    /// Keys whose entry was added, replaced or removed, grouped by publication group.
    pub groups: Vec<Vec<BrickKey>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub dirtied: u64,
    pub coalesced: u64,
    pub overflow_deferred: u64,
    pub max_queue: usize,
    pub max_overflow: usize,
    pub groups_merged: u64,
    pub max_group_len: usize,
    pub dispatched: u64,
    pub accepted: u64,
    /// Accepted although an input version changed, because the relevant content did not.
    pub accepted_by_content: u64,
    pub rejected_cancelled: u64,
    pub rejected_stale: u64,
    pub rejected_stale_at_publish: u64,
    pub rejected_unknown: u64,
    pub staged_discarded: u64,
    pub publish_deferred: u64,
    pub published: u64,
    pub groups_published: u64,
    pub retired: u64,
    pub freed: u64,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    handle: Handle,
    record: InputRecord,
}

#[derive(Clone, Debug)]
struct Staged {
    handle: Option<Handle>,
    deps: Dependencies,
    /// World version at which `deps` was last found valid. Revalidation is needed only after the
    /// world moves on: equal versions mean equal content (the 1C fast path already relies on it).
    validated_at: ContentVersion,
}

/// One chunk's entries. Shared by every snapshot in which the chunk did not change.
type Shard = BTreeMap<BrickKey, Entry>;

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub id: SnapshotId,
    /// World version at publication.
    pub world_version: ContentVersion,
    /// Non-empty shards only.
    shards: BTreeMap<ChunkCoord, Arc<Shard>>,
    len: usize,
}

impl Snapshot {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Keys in `BrickKey` order (chunk first, as `BrickKey` orders).
    pub fn keys(&self) -> impl Iterator<Item = BrickKey> + '_ {
        self.entries().map(|(k, _)| k)
    }
    pub fn handle(&self, key: BrickKey) -> Option<Handle> {
        self.entry(key).map(|e| e.handle)
    }
    pub fn record(&self, key: BrickKey) -> Option<InputRecord> {
        self.entry(key).map(|e| e.record)
    }
    /// Number of chunk shards, and how many of them this snapshot shares with `other`.
    pub fn shards_shared_with(&self, other: &Snapshot) -> (usize, usize) {
        let shared = self.shards.iter().filter(|(c, s)| other.shards.get(c).is_some_and(|o| Arc::ptr_eq(s, o))).count();
        (self.shards.len(), shared)
    }
    fn entry(&self, key: BrickKey) -> Option<&Entry> {
        self.shards.get(&key.chunk).and_then(|s| s.get(&key))
    }
    fn entries(&self) -> impl Iterator<Item = (BrickKey, &Entry)> + '_ {
        self.shards.values().flat_map(|s| s.iter().map(|(&k, e)| (k, e)))
    }
}

/// A reader's claim on one snapshot, like a GPU frame in flight. Release it explicitly.
#[derive(Debug, PartialEq, Eq)]
pub struct ReaderToken {
    pub id: ReaderId,
    pub snapshot: SnapshotId,
}

#[derive(Debug)]
struct InFlight {
    key: BrickKey,
    cancelled: bool,
}

#[derive(Debug)]
pub struct Pipeline {
    cfg: Config,
    queue: VecDeque<BrickKey>,
    queued: BTreeSet<BrickKey>,
    overflow: BTreeSet<BrickKey>,
    in_flight: BTreeMap<JobId, InFlight>,
    running: BTreeMap<BrickKey, JobId>,
    /// Members of unpublished groups; each member is either pending or staged.
    groups: BTreeMap<GroupId, BTreeSet<BrickKey>>,
    group_of: BTreeMap<BrickKey, GroupId>,
    pending: BTreeSet<BrickKey>,
    staged: BTreeMap<BrickKey, Staged>,
    /// Groups that had a member staged since the last `try_publish`. Only these can have become
    /// ready: dirtying a key always un-stages it, so readiness arises only in `complete`.
    touched: BTreeSet<GroupId>,
    pool: SlotPool<BrickMesh>,
    snapshots: BTreeMap<SnapshotId, Snapshot>,
    current: SnapshotId,
    readers: BTreeMap<ReaderId, SnapshotId>,
    /// (last snapshot that contains the handle, handle)
    retiring: Vec<(SnapshotId, Handle)>,
    next_job: u64,
    next_group: u64,
    next_snapshot: u64,
    next_reader: u64,
    stats: Stats,
}

impl Pipeline {
    pub fn new(cfg: Config) -> Self {
        assert!(cfg.queue_capacity > 0 && cfg.max_in_flight > 0, "queue and in-flight limits must be positive");
        let first = SnapshotId(0);
        let mut snapshots = BTreeMap::new();
        snapshots.insert(first, Snapshot { id: first, world_version: ContentVersion::NONE, shards: BTreeMap::new(), len: 0 });
        Self {
            cfg,
            queue: VecDeque::new(),
            queued: BTreeSet::new(),
            overflow: BTreeSet::new(),
            in_flight: BTreeMap::new(),
            running: BTreeMap::new(),
            groups: BTreeMap::new(),
            group_of: BTreeMap::new(),
            pending: BTreeSet::new(),
            staged: BTreeMap::new(),
            touched: BTreeSet::new(),
            pool: SlotPool::new(),
            snapshots,
            current: first,
            readers: BTreeMap::new(),
            retiring: Vec::new(),
            next_job: 0,
            next_group: 0,
            next_snapshot: 1,
            next_reader: 0,
            stats: Stats::default(),
        }
    }

    // ---- dirty tracking and queueing

    /// One batch covering every brick currently in the world and every key in the current snapshot.
    pub fn mark_all(&mut self, world: &World) {
        let keys: BTreeSet<BrickKey> = world.bricks().map(|(k, _)| k).chain(self.snapshots[&self.current].keys()).collect();
        self.mark_batch(keys);
    }

    /// Call after changing these voxels in the world. The whole call is one edit batch.
    pub fn notify_edits(&mut self, changed: &[VoxelCoord]) {
        let mut keys = BTreeSet::new();
        for &v in changed {
            if self.cfg.faults.skip_neighbour_dirty {
                keys.insert(v.split().0);
            } else {
                keys.extend(affected_keys(v));
            }
        }
        self.mark_batch(keys);
    }

    /// Marks `keys` pending as one publication group, merged with any unpublished group they touch.
    pub fn mark_batch(&mut self, keys: BTreeSet<BrickKey>) {
        if keys.is_empty() {
            return;
        }
        let touched: BTreeSet<GroupId> = keys.iter().filter_map(|k| self.group_of.get(k).copied()).collect();
        self.stats.groups_merged += touched.len() as u64;
        let g = GroupId(self.next_group);
        self.next_group += 1;
        let mut members = keys.clone();
        for t in touched {
            members.extend(self.groups.remove(&t).expect("group of a member exists"));
        }
        for &k in &members {
            self.group_of.insert(k, g);
        }
        self.stats.max_group_len = self.stats.max_group_len.max(members.len());
        self.groups.insert(g, members);
        for k in keys {
            self.mark_dirty(k);
        }
    }

    fn mark_dirty(&mut self, key: BrickKey) {
        self.stats.dirtied += 1;
        self.pending.insert(key);
        if !self.cfg.faults.skip_cancel {
            if let Some(id) = self.running.remove(&key) {
                self.in_flight.get_mut(&id).expect("running job is in flight").cancelled = true;
            }
        }
        if let Some(s) = self.staged.remove(&key) {
            // Never published, so no reader can hold it: free immediately.
            self.stats.staged_discarded += 1;
            if let Some(h) = s.handle {
                self.pool.free(h).expect("staged handle is live");
            }
        }
        self.enqueue(key);
    }

    fn enqueue(&mut self, key: BrickKey) {
        if self.queued.contains(&key) || self.overflow.contains(&key) {
            self.stats.coalesced += 1;
        } else if self.queue.len() < self.cfg.queue_capacity {
            self.queue.push_back(key);
            self.queued.insert(key);
            self.stats.max_queue = self.stats.max_queue.max(self.queue.len());
        } else {
            self.overflow.insert(key);
            self.stats.overflow_deferred += 1;
            self.stats.max_overflow = self.stats.max_overflow.max(self.overflow.len());
        }
    }

    fn refill(&mut self) {
        while self.queue.len() < self.cfg.queue_capacity {
            let Some(k) = self.overflow.pop_first() else { break };
            self.queue.push_back(k);
            self.queued.insert(k);
        }
    }

    /// Dispatches queued keys up to the in-flight limit. Inputs are captured now, from `world`.
    pub fn dispatch(&mut self, world: &World) -> Vec<Job> {
        let mut jobs = Vec::new();
        self.refill();
        while self.in_flight.len() < self.cfg.max_in_flight {
            let Some(key) = self.queue.pop_front() else { break };
            self.queued.remove(&key);
            self.refill();
            if !self.pending.contains(&key) {
                // Only reachable with planted faults that accept a result while the key is queued.
                continue;
            }
            let id = JobId(self.next_job);
            self.next_job += 1;
            self.in_flight.insert(id, InFlight { key, cancelled: false });
            if let Some(prev) = self.running.insert(key, id) {
                debug_assert!(self.cfg.faults.skip_cancel, "{prev:?} should have been cancelled");
            }
            self.stats.dispatched += 1;
            jobs.push(Job { id, key, merge: self.cfg.merge, inputs: InputSnapshot::capture(world, key) });
        }
        jobs
    }

    /// Hands back a finished job. `world` must be the current world.
    pub fn complete(&mut self, world: &World, result: JobResult) -> Completion {
        let Some(job) = self.in_flight.remove(&result.id) else {
            self.stats.rejected_unknown += 1;
            return Completion::Rejected(Rejection::UnknownJob);
        };
        debug_assert_eq!(job.key, result.key);
        if self.running.get(&job.key) == Some(&result.id) {
            self.running.remove(&job.key);
        }
        if job.cancelled {
            // mark_dirty already queued the key again.
            self.stats.rejected_cancelled += 1;
            return Completion::Rejected(Rejection::Cancelled);
        }
        if !self.cfg.faults.skip_stale_check {
            if let Some(changed) = result.deps.first_changed(world) {
                self.stats.rejected_stale += 1;
                if !self.running.contains_key(&job.key) {
                    self.enqueue(job.key);
                }
                return Completion::Rejected(Rejection::Stale { changed });
            }
            if result.deps.record.first_version_change(world).is_some() {
                self.stats.accepted_by_content += 1;
            }
        }
        if !self.group_of.contains_key(&job.key) {
            // Only reachable with planted faults: the key's group was already published without it.
            self.stats.rejected_cancelled += 1;
            return Completion::Rejected(Rejection::Cancelled);
        }
        let handle = result.output.map(|s| self.pool.alloc(s));
        self.touched.insert(self.group_of[&job.key]);
        if let Some(old) = self.staged.insert(job.key, Staged { handle, deps: result.deps, validated_at: world.version() }) {
            self.stats.staged_discarded += 1;
            if let Some(h) = old.handle {
                self.pool.free(h).expect("staged handle is live");
            }
        }
        // With `skip_cancel` a newer job for the key may still be out; the key stays pending for it.
        if !self.running.contains_key(&job.key) {
            self.pending.remove(&job.key);
        }
        self.stats.accepted += 1;
        if self.cfg.faults.publish_partial_groups {
            self.pending.remove(&job.key);
            let g = self.group_of[&job.key];
            self.split_off(g, job.key);
        }
        Completion::Accepted
    }

    /// Fault support: moves `key` into its own group so it can publish alone.
    fn split_off(&mut self, g: GroupId, key: BrickKey) {
        let members = self.groups.get_mut(&g).expect("group exists");
        if members.len() == 1 {
            return;
        }
        members.remove(&key);
        let solo = GroupId(self.next_group);
        self.next_group += 1;
        self.groups.insert(solo, BTreeSet::from([key]));
        self.group_of.insert(key, solo);
        self.touched.insert(solo);
    }

    // ---- publication

    /// Publishes every group whose members are all staged, as one new snapshot.
    pub fn try_publish(&mut self, world: &World) -> Option<Published> {
        if !self.cfg.faults.skip_stale_check {
            let now = world.version();
            let mut stale = Vec::new();
            for (&k, s) in self.staged.iter_mut() {
                if s.validated_at == now {
                    continue;
                }
                if s.deps.first_changed(world).is_some() {
                    stale.push(k);
                } else {
                    s.validated_at = now;
                }
            }
            for k in stale {
                self.stats.rejected_stale_at_publish += 1;
                let s = self.staged.remove(&k).expect("listed");
                if let Some(h) = s.handle {
                    self.pool.free(h).expect("staged handle is live");
                }
                self.pending.insert(k);
                if !self.running.contains_key(&k) {
                    self.enqueue(k);
                }
            }
        }
        let candidates = std::mem::take(&mut self.touched);
        let ready: Vec<GroupId> = candidates.into_iter().filter(|g| self.groups.get(g).is_some_and(|m| m.iter().all(|k| self.staged.contains_key(k)))).collect();
        if ready.is_empty() {
            if !self.groups.is_empty() {
                self.stats.publish_deferred += 1;
            }
            return None;
        }
        let prev = self.current;
        // Copies the chunk index only; `make_mut` copies a shard the first time it changes here.
        let mut shards = self.snapshots[&prev].shards.clone();
        let mut len = self.snapshots[&prev].len;
        let mut published_groups = Vec::new();
        for g in ready {
            let members = self.groups.remove(&g).expect("ready group exists");
            for &k in &members {
                self.group_of.remove(&k);
                let s = self.staged.remove(&k).expect("ready members are staged");
                let old = shards.get_mut(&k.chunk).and_then(|sh| if sh.contains_key(&k) { Arc::make_mut(sh).remove(&k) } else { None });
                if let Some(old) = old {
                    len -= 1;
                    self.retiring.push((prev, old.handle));
                    self.stats.retired += 1;
                }
                if let Some(h) = s.handle {
                    Arc::make_mut(shards.entry(k.chunk).or_default()).insert(k, Entry { handle: h, record: s.deps.record });
                    len += 1;
                }
                if shards.get(&k.chunk).is_some_and(|sh| sh.is_empty()) {
                    shards.remove(&k.chunk);
                }
            }
            self.stats.groups_published += 1;
            published_groups.push(members.into_iter().collect());
        }
        let id = SnapshotId(self.next_snapshot);
        self.next_snapshot += 1;
        self.snapshots.insert(id, Snapshot { id, world_version: world.version(), shards, len });
        self.current = id;
        self.stats.published += 1;
        if self.cfg.faults.retire_without_readers {
            for (_, h) in std::mem::take(&mut self.retiring) {
                self.pool.free(h);
                self.stats.freed += 1;
            }
        }
        self.collect();
        Some(Published { id, groups: published_groups })
    }

    pub fn current(&self) -> &Snapshot {
        &self.snapshots[&self.current]
    }

    /// True while `key` belongs to an unpublished group (its published value may be older than the world).
    pub fn is_awaiting(&self, key: BrickKey) -> bool {
        self.group_of.contains_key(&key)
    }

    // ---- readers and retirement

    pub fn acquire(&mut self) -> ReaderToken {
        let id = ReaderId(self.next_reader);
        self.next_reader += 1;
        self.readers.insert(id, self.current);
        ReaderToken { id, snapshot: self.current }
    }

    pub fn release(&mut self, token: ReaderToken) -> Result<(), ReadError> {
        self.readers.remove(&token.id).ok_or(ReadError::UnknownReader)?;
        self.collect();
        Ok(())
    }

    /// Reads `key` from the reader's snapshot. `Ok(None)` means the brick is absent (empty) in that snapshot.
    pub fn read(&self, token: &ReaderToken, key: BrickKey) -> Result<Option<&BrickMesh>, ReadError> {
        let snap = self.readers.get(&token.id).and_then(|s| self.snapshots.get(s)).ok_or(ReadError::UnknownReader)?;
        match snap.entry(key) {
            None => Ok(None),
            Some(e) => self.pool.get(e.handle).map(Some).ok_or(ReadError::Dangling { key }),
        }
    }

    /// Resolves a handle directly; used by tests to show that retired handles stop resolving.
    pub fn resolve(&self, h: Handle) -> Option<&BrickMesh> {
        self.pool.get(h)
    }

    /// Frees retired resources that no live reader can see, and drops unreferenced old snapshots.
    fn collect(&mut self) {
        let oldest_live = self.readers.values().copied().min().unwrap_or(self.current).min(self.current);
        let pool = &mut self.pool;
        let stats = &mut self.stats;
        self.retiring.retain(|&(last, h)| {
            if last < oldest_live {
                pool.free(h).expect("retiring handle is live");
                stats.freed += 1;
                false
            } else {
                true
            }
        });
        let live: BTreeSet<SnapshotId> = self.readers.values().copied().chain([self.current]).collect();
        self.snapshots.retain(|id, _| live.contains(id));
    }

    // ---- introspection

    /// Checks that every entry of every live snapshot resolves. A violation is a retirement bug.
    pub fn check_live_handles(&self) -> Result<(), String> {
        for s in self.snapshots.values() {
            for (k, e) in s.entries() {
                if self.pool.get(e.handle).is_none() {
                    return Err(format!("snapshot {:?} entry {k:?} dangles (handle {:?})", s.id, e.handle));
                }
            }
        }
        Ok(())
    }

    /// Heap held by the pipeline, by category (Phase 1D). Every snapshot alive at once is counted:
    /// the current one and every one kept for a reader. Dispatched jobs are held by the caller
    /// (see [`Job::heap`]).
    pub fn memory(&self) -> memory::Report {
        use memory::{containers as c, Category};
        let mut r = memory::Report::new();
        r.add(Category::DerivedData, self.pool.memory(BrickMesh::heap));

        // Shared shards are counted once (by identity) over every live snapshot.
        let mut snaps = c::btree_map(&self.snapshots);
        let mut seen = BTreeSet::new();
        for s in self.snapshots.values() {
            snaps += c::btree_map(&s.shards);
            for sh in s.shards.values() {
                if seen.insert(Arc::as_ptr(sh)) {
                    snaps += memory::Usage::exact(std::mem::size_of::<Shard>() as u64 + 2 * std::mem::size_of::<usize>() as u64) + c::btree_map(sh);
                }
            }
        }
        r.add(Category::DerivedSnapshots, snaps);

        let mut book = c::vec_deque(&self.queue)
            + c::btree_set(&self.queued)
            + c::btree_set(&self.overflow)
            + c::btree_map(&self.in_flight)
            + c::btree_map(&self.running)
            + c::btree_map(&self.groups)
            + c::btree_map(&self.group_of)
            + c::btree_set(&self.pending)
            + c::btree_map(&self.staged)
            + c::btree_set(&self.touched)
            + c::btree_map(&self.readers)
            + c::vec(&self.retiring);
        for g in self.groups.values() {
            book += c::btree_set(g);
        }
        for s in self.staged.values() {
            book += s.deps.heap();
        }
        r.add(Category::DerivedBookkeeping, book);
        r
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn pool_stats(&self) -> PoolStats {
        self.pool.stats()
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    pub fn overflow_len(&self) -> usize {
        self.overflow.len()
    }

    pub fn live_snapshots(&self) -> usize {
        self.snapshots.len()
    }

    pub fn retiring_len(&self) -> usize {
        self.retiring.len()
    }

    /// Nothing queued, running, pending, staged or awaiting publication.
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.overflow.is_empty() && self.in_flight.is_empty() && self.pending.is_empty() && self.staged.is_empty() && self.groups.is_empty()
    }
}
