//! Phase 1 CPU performance baseline on the street block. It records costs; it does not judge them.
//!
//! Usage: `perf_phase1 [--rounds N]`. Prints one JSON object per line, each with every raw sample
//! (milliseconds unless named otherwise) and its median. Arms that are compared (worker counts,
//! edit sizes) are interleaved round by round, so drift (thermal, background load) spreads over
//! all arms instead of biasing one. The first round is cold and is kept in the raw lists.
//!
//! Correctness is checked outside the timed sections: every published snapshot must equal a
//! from-scratch recomputation.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use derived::{extract_world, Config, Job, JobResult, Merge, Pipeline};
use world::reference::{trace, Ray};
use world::{scene, Transaction, VoxelCoord, World};

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn list(v: &[f64]) -> String {
    format!("[{}]", v.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(","))
}

fn series(name: &str, v: &[f64]) -> String {
    format!("\"{name}\":{},\"{name}_median\":{:.3}", list(v), median(v))
}

/// Every stored brick's published summary equals a recomputation, and nothing else is published.
fn verify(p: &Pipeline, w: &World) {
    let snap = p.current();
    assert_eq!(snap.len(), w.bricks().count(), "published key count");
    for (k, _) in w.bricks() {
        let got = p.resolve(snap.handle(k).expect("key published")).expect("handle resolves");
        assert_eq!(Some(got), extract_world(w, k, Merge::default()).as_ref(), "stale entry {k:?}");
    }
}

struct Workers {
    jobs: Option<mpsc::Sender<Job>>,
    results: mpsc::Receiver<JobResult>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl Workers {
    fn new(n: usize) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (res_tx, res_rx) = mpsc::channel::<JobResult>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let handles = (0..n)
            .map(|_| {
                let rx = Arc::clone(&job_rx);
                let tx = res_tx.clone();
                thread::spawn(move || loop {
                    let job = match rx.lock().unwrap().recv() {
                        Ok(j) => j,
                        Err(_) => break,
                    };
                    if tx.send(job.run()).is_err() {
                        break;
                    }
                })
            })
            .collect();
        Self { jobs: Some(job_tx), results: res_rx, handles }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.jobs.take();
        for h in self.handles.drain(..) {
            h.join().unwrap();
        }
    }
}

/// Runs the pipeline until idle. `workers == None` runs every job inline on this thread.
/// Returns (jobs run, milliseconds spent in `try_publish`).
fn drain(p: &mut Pipeline, w: &World, workers: Option<&Workers>) -> (usize, f64) {
    let (n, publish_ms, _) = drain_counted(p, w, workers);
    (n, publish_ms)
}

/// As [`drain`], also returning the number of `try_publish` calls.
fn drain_counted(p: &mut Pipeline, w: &World, workers: Option<&Workers>) -> (usize, f64, usize) {
    let mut n = 0;
    let mut publish_ms = 0.0;
    let mut calls = 0;
    loop {
        let jobs = p.dispatch(w);
        let dispatched = jobs.len();
        match workers {
            None => {
                for j in jobs {
                    p.complete(w, j.run());
                }
            }
            Some(ws) => {
                for j in jobs {
                    ws.jobs.as_ref().unwrap().send(j).unwrap();
                }
                for _ in 0..dispatched {
                    let r = ws.results.recv().unwrap();
                    p.complete(w, r);
                }
            }
        }
        n += dispatched;
        let t = Instant::now();
        p.try_publish(w);
        publish_ms += ms(t);
        calls += 1;
        if dispatched == 0 && p.is_idle() {
            break;
        }
    }
    (n, publish_ms, calls)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.iter().position(|a| a == "--rounds").map(|i| args[i + 1].parse().unwrap()).unwrap_or(5);
    let threads_available = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("{{\"section\":\"environment\",\"rounds\":{rounds},\"available_parallelism\":{threads_available},\"profile\":\"{}\"}}", if cfg!(debug_assertions) { "debug" } else { "release" });

    // 1. Scene generation.
    let mut gen = Vec::new();
    for _ in 0..rounds {
        let t = Instant::now();
        let (w, _) = scene::street_block();
        gen.push(ms(t));
        drop(w);
    }
    let (world, _) = scene::street_block();
    let s = world.stats();
    println!("{{\"section\":\"generate\",\"bricks\":{},\"voxels\":{},{}}}", s.bricks, s.occupied_voxels, series("ms", &gen));

    // 2. World edits: a 32^3 fill above the scene and its clear, as transactions.
    let glass = world.materials().id_of("glass").unwrap();
    let (mut fill, mut clear) = (Vec::new(), Vec::new());
    let mut changed = 0;
    for _ in 0..rounds {
        let mut w = world.clone();
        let (lo, hi) = (VoxelCoord::new(0, 170, 0), VoxelCoord::new(32, 202, 32));
        let mut tx = Transaction::new();
        tx.fill(lo, hi, Some(glass));
        let t = Instant::now();
        changed = w.apply(&tx).unwrap().changed.len();
        fill.push(ms(t));
        let mut tx = Transaction::new();
        tx.fill(lo, hi, None);
        let t = Instant::now();
        w.apply(&tx).unwrap();
        clear.push(ms(t));
    }
    println!("{{\"section\":\"world_edit_32cube\",\"voxels\":{changed},{},{}}}", series("fill_ms", &fill), series("clear_ms", &clear));

    // 3. Full derived build with 0 (inline), 1, 2, 4, 8 workers, interleaved.
    let arms = [0usize, 1, 2, 4, 8];
    let mut build: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut publish: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut jobs_run = 0;
    for _ in 0..rounds {
        for (i, &n) in arms.iter().enumerate() {
            let ws = (n > 0).then(|| Workers::new(n));
            let mut p = Pipeline::new(Config::default());
            let t = Instant::now();
            p.mark_all(&world);
            let (jobs, pub_ms) = drain(&mut p, &world, ws.as_ref());
            build[i].push(ms(t));
            publish[i].push(pub_ms);
            jobs_run = jobs;
            verify(&p, &world);
        }
    }
    for (i, &n) in arms.iter().enumerate() {
        println!(
            "{{\"section\":\"derived_full_build\",\"workers\":{n},\"jobs\":{jobs_run},{},{},\"jobs_per_s_median\":{:.0}}}",
            series("ms", &build[i]),
            series("publish_ms", &publish[i]),
            jobs_run as f64 / (median(&build[i]) / 1e3)
        );
    }

    // 3a. Surface extraction per merge mode (Phase 2A): mesh size, and the inline full build, interleaved.
    // GPU bytes are a projection from the ADR-0004 layout (64 B per quad), not a measurement.
    let mut mesh_ms: Vec<Vec<f64>> = vec![Vec::new(); Merge::ALL.len()];
    let mut mesh_counts = vec![(0usize, 0usize, 0usize); Merge::ALL.len()];
    for _ in 0..rounds {
        for (i, &merge) in Merge::ALL.iter().enumerate() {
            let mut p = Pipeline::new(Config { merge, ..Config::default() });
            let t = Instant::now();
            p.mark_all(&world);
            drain(&mut p, &world, None);
            mesh_ms[i].push(ms(t));
            let snap = p.current();
            let sizes: Vec<usize> = snap.keys().map(|k| p.resolve(snap.handle(k).unwrap()).unwrap().quads.len()).collect();
            mesh_counts[i] = (sizes.iter().sum(), sizes.iter().copied().max().unwrap_or(0), sizes.iter().filter(|&&n| n == 0).count());
            for k in snap.keys() {
                assert_eq!(p.resolve(snap.handle(k).unwrap()), extract_world(&world, k, merge).as_ref(), "stale entry {k:?}");
            }
        }
    }
    for (i, &merge) in Merge::ALL.iter().enumerate() {
        let (quads, max, hidden) = mesh_counts[i];
        println!(
            "{{\"section\":\"mesh\",\"merge\":\"{}\",\"bricks\":{},\"quads\":{quads},\"triangles\":{},\"max_quads_per_brick\":{max},\"bricks_without_quads\":{hidden},\"cpu_quad_bytes\":{},\"gpu_geometry_bytes_projected\":{},{}}}",
            merge.name(),
            s.bricks,
            quads * 2,
            quads * 8,
            quads * 64,
            series("full_build_inline_ms", &mesh_ms[i])
        );
    }

    // 3b. Publication cost against the number of `try_publish` calls (inline full build, in-flight
    // limit varied, interleaved). If publish time is proportional to calls, each call rescans all staged work.
    let limits = [16usize, 64, 256, 1024];
    let mut pub_by: Vec<Vec<f64>> = vec![Vec::new(); limits.len()];
    let mut calls_by = vec![0usize; limits.len()];
    for _ in 0..rounds {
        for (i, &lim) in limits.iter().enumerate() {
            let mut p = Pipeline::new(Config { queue_capacity: lim.max(256), max_in_flight: lim, ..Config::default() });
            p.mark_all(&world);
            let (_, pub_ms, calls) = drain_counted(&mut p, &world, None);
            pub_by[i].push(pub_ms);
            calls_by[i] = calls;
            verify(&p, &world);
        }
    }
    for (i, &lim) in limits.iter().enumerate() {
        println!(
            "{{\"section\":\"publish_scaling\",\"max_in_flight\":{lim},\"publish_calls\":{},{},\"ms_per_call_median\":{:.4}}}",
            calls_by[i],
            series("publish_ms", &pub_by[i]),
            median(&pub_by[i]) / calls_by[i] as f64
        );
    }

    // 4. Edit-to-publish latency on the CPU (inline jobs): one voxel, an 8^3 box straddling 8 bricks,
    // and a 32^3 box, interleaved. Each edit alternates between two materials so it always changes content.
    let asphalt = world.materials().id_of("asphalt").unwrap();
    let kinds: [(&str, VoxelCoord, VoxelCoord); 3] = [
        ("voxel", VoxelCoord::new(100, 4, 60), VoxelCoord::new(101, 5, 61)),
        ("box8", VoxelCoord::new(100, 4, 60), VoxelCoord::new(108, 12, 68)),
        ("box32", VoxelCoord::new(96, 0, 56), VoxelCoord::new(128, 32, 88)),
    ];
    let mut w = world.clone();
    let mut p = Pipeline::new(Config::default());
    p.mark_all(&w);
    drain(&mut p, &w, None);
    let per = rounds * 4;
    let mut stage: Vec<[Vec<f64>; 5]> = (0..kinds.len()).map(|_| Default::default()).collect();
    let mut sizes = vec![(0usize, 0usize); kinds.len()];
    for r in 0..per {
        for (i, &(_, lo, hi)) in kinds.iter().enumerate() {
            let mut tx = Transaction::new();
            tx.fill(lo, hi, Some(if r % 2 == 0 { glass } else { asphalt }));
            let t0 = Instant::now();
            let applied = w.apply(&tx).unwrap();
            let t_apply = ms(t0);
            let t1 = Instant::now();
            p.notify_edits(&applied.changed);
            let t_notify = ms(t1);
            let t2 = Instant::now();
            let (jobs, pub_ms) = drain(&mut p, &w, None);
            let t_jobs = ms(t2) - pub_ms;
            stage[i][0].push(t_apply);
            stage[i][1].push(t_notify);
            stage[i][2].push(t_jobs);
            stage[i][3].push(pub_ms);
            stage[i][4].push(t_apply + t_notify + t_jobs + pub_ms);
            sizes[i] = (applied.changed.len(), jobs);
        }
        if r == 0 || r == per - 1 {
            verify(&p, &w);
        }
    }
    for (i, &(name, _, _)) in kinds.iter().enumerate() {
        let [a, n, j, pb, tot] = &stage[i];
        println!(
            "{{\"section\":\"edit_to_publish_cpu\",\"edit\":\"{name}\",\"changed_voxels\":{},\"jobs\":{},{},{},{},{},{}}}",
            sizes[i].0,
            sizes[i].1,
            series("apply_ms", a),
            series("notify_ms", n),
            series("jobs_ms", j),
            series("publish_ms", pb),
            series("total_ms", tot)
        );
    }

    // 5. Reference ray query throughput (single thread): rays from above the street toward random ground points.
    let (lo, hi) = world.bounds().unwrap();
    let mut rng = Rng(0x5EED);
    let n_rays = 200_000;
    let rays: Vec<Ray> = (0..n_rays)
        .map(|_| {
            let o = [lo.x as f64 + rng.unit() * (hi.x - lo.x) as f64, hi.y as f64 + 8.0, lo.z as f64 + rng.unit() * (hi.z - lo.z) as f64];
            let d = [rng.unit() - 0.5, -1.0, rng.unit() - 0.5];
            Ray { origin: o, dir: d }
        })
        .collect();
    let mut trace_ms = Vec::new();
    let mut hits = 0;
    for _ in 0..rounds {
        let t = Instant::now();
        hits = rays.iter().filter(|r| trace(&world, r, 1e6).is_some()).count();
        trace_ms.push(ms(t));
    }
    println!(
        "{{\"section\":\"reference_trace\",\"rays\":{n_rays},\"hits\":{hits},{},\"mrays_per_s_median\":{:.3}}}",
        series("ms", &trace_ms),
        n_rays as f64 / (median(&trace_ms) / 1e3) / 1e6
    );

    // 6. Memory at the end of the edit sequence (reported, 1D).
    println!("{{\"section\":\"memory\",\"world\":{},\"pipeline\":{}}}", w.memory().to_json(), p.memory().to_json());
}
