//! Scripted Phase 1C scenarios: each controls completion order and timing explicitly.

mod common;

use common::{check_snapshot, run_to_idle, truth, world_with};
use derived::{Completion, Config, Faults, Pipeline, ReadError, Rejection};
use world::{VoxelCoord, World};

fn v(x: i32, y: i32, z: i32) -> VoxelCoord {
    VoxelCoord::new(x, y, z)
}

/// Brick A at voxels [0, 8), brick B at [8, 16) along x, both partly filled; plus far brick C.
fn two_bricks() -> (World, Vec<world::MaterialId>) {
    let (mut w, m) = world_with(3);
    w.fill_box(v(2, 2, 2), v(8, 5, 5), Some(m[0])).unwrap(); // touches A's +x face (x = 7)
    w.fill_box(v(8, 2, 2), v(12, 5, 5), Some(m[1])).unwrap(); // touches B's -x face (x = 8)
    w.fill_box(v(40, 0, 0), v(42, 2, 2), Some(m[2])).unwrap();
    (w, m)
}

fn populated() -> (World, Vec<world::MaterialId>, Pipeline) {
    let (w, m) = two_bricks();
    let mut p = Pipeline::new(Config::default());
    p.mark_all(&w);
    run_to_idle(&mut p, &w);
    check_snapshot(&p, &w).unwrap();
    (w, m, p)
}

fn edit(p: &mut Pipeline, w: &mut World, at: VoxelCoord, m: Option<world::MaterialId>) {
    assert!(w.set(at, m).unwrap(), "edit must change the world");
    p.notify_edits(&[at]);
}

#[test]
fn initial_population_matches_oracle() {
    let (w, _, p) = populated();
    assert_eq!(p.current().len(), 3);
    assert_eq!(p.stats().accepted, 3);
    assert_eq!(p.current().world_version, w.version());
}

#[test]
fn redirtied_target_cancels_the_in_flight_job() {
    let (mut w, m, mut p) = populated();
    edit(&mut p, &mut w, v(3, 3, 3), None); // interior of A
    let old = p.dispatch(&w);
    assert_eq!(old.len(), 1);
    edit(&mut p, &mut w, v(3, 3, 3), Some(m[2])); // A again, while its job is out
    let new = p.dispatch(&w);
    assert_eq!(new.len(), 1);
    // The old job finishes last; it must not overwrite the newer result.
    assert_eq!(p.complete(&w, new.into_iter().next().unwrap().run()), Completion::Accepted);
    assert_eq!(p.complete(&w, old.into_iter().next().unwrap().run()), Completion::Rejected(Rejection::Cancelled));
    assert!(p.try_publish(&w).is_some());
    check_snapshot(&p, &w).unwrap();
}

#[test]
fn neighbour_interior_change_is_accepted_by_content() {
    let (mut w, m, mut p) = populated();
    edit(&mut p, &mut w, v(3, 3, 3), None);
    let job = p.dispatch(&w).pop().unwrap();
    // An interior edit in B changes B's version, but not the face layer A depends on.
    edit(&mut p, &mut w, v(10, 3, 3), Some(m[2]));
    assert_eq!(p.complete(&w, job.run()), Completion::Accepted);
    assert_eq!(p.stats().accepted_by_content, 1);
    run_to_idle(&mut p, &w);
    check_snapshot(&p, &w).unwrap();
}

#[test]
fn unnotified_face_change_is_rejected_as_stale() {
    let (mut w, m, mut p) = populated();
    let b = v(9, 3, 3).split().0;
    edit(&mut p, &mut w, v(3, 3, 3), None);
    let job = p.dispatch(&w).pop().unwrap();
    // The world changes on the A/B face, but the notification has not arrived yet when A's job completes.
    assert!(w.set(v(8, 3, 3), Some(m[2])).unwrap());
    assert_eq!(p.complete(&w, job.run()), Completion::Rejected(Rejection::Stale { changed: b }));
    p.notify_edits(&[v(8, 3, 3)]);
    run_to_idle(&mut p, &w);
    check_snapshot(&p, &w).unwrap();
    assert_eq!(p.stats().rejected_stale, 1);
}

/// A result staged while its group waits must be revalidated at publication once the world has
/// moved on, even though it was valid when it completed. This is the only path that exercises
/// publish-time staleness (`rejected_stale_at_publish`), including after the revalidation skip.
#[test]
fn staged_result_is_revalidated_at_publication_after_an_unnotified_change() {
    let (mut w, _, mut p) = populated();
    let (a, c) = (v(3, 3, 3).split().0, v(41, 1, 1).split().0);
    // One batch: A (voxel 3,3,3) and far brick C. Both keys in one group; B is not dirtied.
    let (va, vc) = (v(3, 3, 3), v(41, 1, 1));
    w.set(va, None).unwrap();
    w.set(vc, None).unwrap();
    p.notify_edits(&[va, vc]);
    let mut jobs = p.dispatch(&w);
    assert_eq!(jobs.len(), 2);
    let job_c = jobs.remove(jobs.iter().position(|j| j.key == c).unwrap());
    let job_a = jobs.pop().unwrap();
    assert_eq!(job_a.key, a);
    assert_eq!(p.complete(&w, job_a.run()), Completion::Accepted);
    // Nothing is ready yet: C is still running. Publication is deferred and revalidates A at this version.
    assert!(p.try_publish(&w).is_none());
    // A's brick changes without a notification (it arrives later), then C completes.
    assert!(w.set(v(4, 3, 3), None).unwrap());
    assert_eq!(p.complete(&w, job_c.run()), Completion::Accepted);
    // The group looks complete, but A's staged result is stale: nothing may publish.
    assert!(p.try_publish(&w).is_none(), "a half-valid group must not publish");
    assert_eq!(p.stats().rejected_stale_at_publish, 1);
    assert!(p.is_awaiting(a) && p.is_awaiting(c));
    p.notify_edits(&[v(4, 3, 3)]);
    run_to_idle(&mut p, &w);
    check_snapshot(&p, &w).unwrap();
}

#[test]
fn duplicate_completion_is_rejected() {
    let (mut w, _, mut p) = populated();
    edit(&mut p, &mut w, v(3, 3, 3), None);
    let r = p.dispatch(&w).pop().unwrap().run();
    assert_eq!(p.complete(&w, r.clone()), Completion::Accepted);
    assert_eq!(p.complete(&w, r), Completion::Rejected(Rejection::UnknownJob));
}

#[test]
fn boundary_edit_rederives_the_neighbour_and_removes_empty_bricks() {
    let (mut w, _, mut p) = populated();
    // Clear all of B: A's +x boundary faces become exposed, and B disappears from the world.
    let changed: Vec<_> = (2..5).flat_map(|z| (2..5).flat_map(move |y| (8..12).map(move |x| v(x, y, z)))).collect();
    for &c in &changed {
        w.set(c, None).unwrap();
    }
    p.notify_edits(&changed);
    run_to_idle(&mut p, &w);
    check_snapshot(&p, &w).unwrap();
    assert_eq!(p.current().len(), 2);
}

#[test]
fn negative_control_missing_neighbour_dirty_is_caught() {
    let (w0, _) = two_bricks();
    let mut p = Pipeline::new(Config { faults: Faults { skip_neighbour_dirty: true, ..Faults::default() }, ..Config::default() });
    let mut w = w0;
    p.mark_all(&w);
    run_to_idle(&mut p, &w);
    // Clearing x = 8 changes A's exposed faces, but with the fault only B is re-derived.
    let changed: Vec<_> = (2..5).flat_map(|z| (2..5).map(move |y| v(8, y, z))).collect();
    for &c in &changed {
        w.set(c, None).unwrap();
    }
    p.notify_edits(&changed);
    run_to_idle(&mut p, &w);
    let err = check_snapshot(&p, &w).expect_err("oracle must detect the stale neighbour");
    assert!(err.contains("stale"), "{err}");
}

#[test]
fn publication_holds_the_prior_snapshot_until_the_whole_set_is_ready() {
    let (mut w, m, mut p) = populated();
    let before = w.clone();
    let first = p.current().id;
    // One edit on the A/B boundary dirties both bricks.
    w.set(v(7, 3, 3), None).unwrap();
    w.set(v(8, 3, 3), Some(m[2])).unwrap();
    p.notify_edits(&[v(7, 3, 3), v(8, 3, 3)]);
    let mut jobs = p.dispatch(&w);
    assert_eq!(jobs.len(), 2);
    let second = jobs.pop().unwrap();
    assert_eq!(p.complete(&w, jobs.pop().unwrap().run()), Completion::Accepted);
    assert_eq!(p.try_publish(&w), None, "half of the set is not publishable");
    assert_eq!(p.current().id, first);
    check_snapshot(&p, &before).expect("readers still see a complete, consistent old snapshot");
    assert_eq!(p.complete(&w, second.run()), Completion::Accepted);
    assert!(p.try_publish(&w).is_some());
    check_snapshot(&p, &w).unwrap();
    assert_eq!(p.stats().publish_deferred, 1);
}

#[test]
fn edit_batches_sharing_a_key_merge_into_one_publication() {
    let (mut w, m, mut p) = populated();
    // Batch 1: the A/B face. Batch 2: B's +x face, which also dirties the next brick along x.
    w.set(v(7, 3, 3), None).unwrap();
    p.notify_edits(&[v(7, 3, 3)]);
    w.set(v(15, 3, 3), Some(m[2])).unwrap();
    p.notify_edits(&[v(15, 3, 3)]);
    assert_eq!(p.stats().groups_merged, 1, "batch 2 touches B, which is in batch 1's group");
    let report = {
        run_to_idle_without_publish(&mut p, &w);
        p.try_publish(&w).expect("merged group is complete")
    };
    assert_eq!(report.groups.len(), 1);
    assert_eq!(report.groups[0].len(), 3, "A, B and B's +x neighbour publish together");
    check_snapshot(&p, &w).unwrap();
}

fn run_to_idle_without_publish(p: &mut Pipeline, w: &World) {
    loop {
        let jobs = p.dispatch(w);
        if jobs.is_empty() {
            break;
        }
        for j in jobs {
            p.complete(w, j.run());
        }
    }
}

#[test]
fn sustained_edits_defer_publication_and_recover() {
    let (mut w, m, mut p) = populated();
    let reader = p.acquire();
    let first = p.current().id;
    let before = w.clone();
    // Every frame: finish all work, then an edit lands on the A/B face before the frame's publish.
    for i in 0..50 {
        run_to_idle_without_publish(&mut p, &w);
        let at = v(7, 3, 3);
        w.set(at, if i % 2 == 0 { None } else { Some(m[0]) }).unwrap();
        p.notify_edits(&[at]);
        assert_eq!(p.try_publish(&w), None);
    }
    // Readers were never exposed to a half-updated state: the old snapshot stays whole and valid.
    assert_eq!(p.current().id, first);
    check_snapshot(&p, &before).unwrap();
    assert_eq!(p.stats().publish_deferred, 50);
    assert!(p.read(&reader, v(3, 3, 3).split().0).unwrap().is_some());
    // Edits stop: the backlog publishes on the next frame.
    run_to_idle(&mut p, &w);
    assert_ne!(p.current().id, first);
    check_snapshot(&p, &w).unwrap();
    p.release(reader).unwrap();
    assert_eq!(p.pool_stats().live, p.current().len());
}

#[test]
fn retirement_waits_for_readers_and_reused_slots_reject_old_handles() {
    let (mut w, m, mut p) = populated();
    let a = v(3, 3, 3).split().0;
    let old_truth = truth(&w)[&a].clone();
    let old_handle = p.current().handle(a).unwrap();
    let reader = p.acquire();

    edit(&mut p, &mut w, v(3, 3, 3), None);
    run_to_idle(&mut p, &w);
    check_snapshot(&p, &w).unwrap();
    // The reader still sees its own snapshot's value, and its resource is still allocated.
    assert_eq!(p.read(&reader, a).unwrap(), Some(&old_truth));
    assert!(p.resolve(old_handle).is_some());
    assert_eq!(p.retiring_len(), 1);
    assert_eq!(p.live_snapshots(), 2);

    p.release(reader).unwrap();
    assert!(p.resolve(old_handle).is_none(), "freed once the last reader left");
    assert_eq!((p.retiring_len(), p.live_snapshots()), (0, 1));

    // Reuse the freed slot: the old handle must still not resolve.
    edit(&mut p, &mut w, v(40, 0, 0), None);
    run_to_idle(&mut p, &w);
    let reused = p.current().keys().filter_map(|k| p.current().handle(k)).find(|h| h.index() == old_handle.index());
    let reused = reused.expect("the lowest free slot is reused");
    assert_ne!(reused.generation(), old_handle.generation());
    assert!(p.resolve(old_handle).is_none());
    assert_eq!(p.pool_stats().live, p.current().len(), "no leaks once readers are gone");
    let _ = m;
}

#[test]
fn negative_control_retiring_without_readers_is_caught() {
    let (mut w, _) = two_bricks();
    let mut p = Pipeline::new(Config { faults: Faults { retire_without_readers: true, ..Faults::default() }, ..Config::default() });
    p.mark_all(&w);
    run_to_idle(&mut p, &w);
    let a = v(3, 3, 3).split().0;
    let reader = p.acquire();
    w.set(v(3, 3, 3), None).unwrap();
    p.notify_edits(&[v(3, 3, 3)]);
    run_to_idle(&mut p, &w);
    assert_eq!(p.read(&reader, a), Err(ReadError::Dangling { key: a }));
    assert!(p.check_live_handles().is_err());
}

#[test]
fn overload_is_bounded_lossless_and_deterministic() {
    let build = || {
        let (mut w, m) = world_with(2);
        // 60 separate bricks.
        for i in 0..60 {
            w.set(v((i % 10) * 8 + 1, (i / 10) * 8 + 1, 1), Some(m[i as usize % 2])).unwrap();
        }
        w
    };
    let run = || {
        let w = build();
        let mut p = Pipeline::new(Config { queue_capacity: 4, max_in_flight: 2, ..Config::default() });
        p.mark_all(&w);
        assert!(p.queue_len() <= 4);
        assert_eq!(p.overflow_len(), 56);
        let mut order = Vec::new();
        loop {
            let jobs = p.dispatch(&w);
            assert!(jobs.len() <= 2 && p.queue_len() <= 4);
            if jobs.is_empty() {
                break;
            }
            // Complete in reverse dispatch order.
            for j in jobs.into_iter().rev() {
                order.push(j.key);
                assert_eq!(p.complete(&w, j.run()), Completion::Accepted);
            }
        }
        assert!(p.try_publish(&w).is_some());
        check_snapshot(&p, &w).unwrap();
        (p.stats(), order)
    };
    let (s1, o1) = run();
    let (s2, o2) = run();
    assert_eq!(s1, s2);
    assert_eq!(o1, o2, "dispatch order is deterministic");
    assert_eq!(s1.accepted, 60);
    assert_eq!(s1.overflow_deferred, 56);
    assert_eq!(s1.max_queue, 4);
}

#[test]
fn publication_shares_unchanged_shards_and_keeps_old_snapshots_intact() {
    // 2E: a publication copies only the chunk shards it changes. A reader of the old snapshot must
    // still see the old values of the changed keys (copy-on-write never writes a shared shard).
    let (mut w, _) = world::scene::street_block();
    let mut p = Pipeline::new(Config::default());
    p.mark_all(&w);
    run_to_idle(&mut p, &w);
    let reader = p.acquire();
    let old = p.current().clone();
    let at = v(100, 4, 60);
    let was = w.get(at);
    let glass = w.materials().id_of("glass").unwrap();
    assert_ne!(was, Some(glass), "the edit must change content");
    edit(&mut p, &mut w, at, Some(glass));
    run_to_idle(&mut p, &w);
    check_snapshot(&p, &w).unwrap();
    let new = p.current().clone();
    let touched: std::collections::BTreeSet<_> = derived::affected_keys(at).into_iter().map(|k| k.chunk).collect();
    let (n, shared) = new.shards_shared_with(&old);
    eprintln!("1-voxel edit: {n} shards, {shared} shared with the previous snapshot, {} chunks touched", touched.len());
    assert!(n - shared <= touched.len(), "only touched chunks are copied: {n} shards, {shared} shared, {} touched", touched.len());
    assert!(shared > 0 && n > 100, "the street has many chunks, nearly all shared");
    let (key, _) = at.split();
    let old_mesh = p.read(&reader, key).unwrap().cloned();
    let now = p.acquire();
    let new_mesh = p.read(&now, key).unwrap().cloned();
    p.release(now).unwrap();
    assert_ne!(old_mesh, new_mesh, "the edited brick changed");
    assert_eq!(p.current().handle(key), new.handle(key));
    assert_ne!(old.handle(key), new.handle(key), "the old snapshot keeps its own entry");
    p.release(reader).unwrap();
    p.check_live_handles().unwrap();
}
