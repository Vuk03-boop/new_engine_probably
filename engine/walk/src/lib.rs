//! Walking collision against the authoritative voxel world (M1, "walking camera").
//!
//! - The walker is an axis-aligned box standing on `feet` (the centre of its bottom face), in voxel
//!   units, Y up. Every occupied voxel is solid: materials carry no collision flag yet.
//! - Motion runs in fixed substeps ([`Params::substep_s`]), so the result does not depend on the
//!   frame rate apart from the last partial substep, which [`Walker::render_eye`] interpolates.
//! - Each substep moves the box one axis at a time (x, then z, then y). An axis move is an exact
//!   sweep: every voxel layer the leading face crosses is tested over the box's whole cross-section,
//!   and the box stops flush against the first solid layer. No speed or substep can tunnel through
//!   a 1-voxel wall or floor.
//! - A grounded walker blocked horizontally tries to step up: up to each whole voxel level within
//!   `step_height`, lowest first, across, then back down. It keeps the first try that got further. A grounded walker that walks off an
//!   edge no higher than `step_height` is snapped down, so curbs do not turn into short falls.
//! - Step-ups and snaps move the camera by a visual offset that decays, not the box.
//! - Cells are tested with a tolerance of [`EPS`] voxels, so a box resting exactly on a face never
//!   counts the voxel behind that face (no snagging on the floor while sliding along a wall).

use world::dims::VOXEL_SIZE_M;
use world::{VoxelCoord, World};

/// Tolerance, in voxels, when turning box faces into voxel ranges.
pub const EPS: f64 = 1e-7;

/// Something the walker collides with.
pub trait Solid {
    fn solid(&self, x: i32, y: i32, z: i32) -> bool;
}

impl Solid for World {
    fn solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.get(VoxelCoord::new(x, y, z)).is_some()
    }
}

/// Walker size and motion. Lengths in voxels, times in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub half_width: f64,
    pub height: f64,
    /// Eye above the feet.
    pub eye_height: f64,
    /// Highest ledge climbed without jumping, and deepest drop snapped down to.
    pub step_height: f64,
    /// Voxels per second squared.
    pub gravity: f64,
    /// Voxels per second.
    pub walk_speed: f64,
    pub run_speed: f64,
    pub jump_speed: f64,
    pub max_fall_speed: f64,
    pub substep_s: f64,
    /// A longer frame (a hitch) is clamped to this, rather than simulated.
    pub max_frame_s: f64,
    /// Decay rate of the step-up/snap camera offset, per second.
    pub eye_settle_per_s: f64,
}

impl Default for Params {
    /// A person: 0.5 m wide, 1.8 m tall, eyes at 1.7 m, steps up to 0.25 m, walks 1.4 m/s, runs
    /// 4 m/s, jumps 0.45 m.
    fn default() -> Self {
        let m = |x: f64| x / VOXEL_SIZE_M;
        let gravity = m(9.81);
        Params {
            half_width: m(0.25),
            height: m(1.8),
            eye_height: m(1.7),
            step_height: m(0.25),
            gravity,
            walk_speed: m(1.4),
            run_speed: m(4.0),
            jump_speed: (2.0 * gravity * m(0.45)).sqrt(),
            max_fall_speed: m(50.0),
            substep_s: 1.0 / 240.0,
            max_frame_s: 0.1,
            eye_settle_per_s: 14.0,
        }
    }
}

/// What the player asks for this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Intent {
    /// Horizontal direction in world x and z; its length (at most 1) scales the speed.
    pub dir: [f64; 2],
    pub run: bool,
    pub jump: bool,
}

/// Counters, for tests and cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub substeps: u64,
    pub steps_up: u64,
    pub snaps_down: u64,
    pub jumps: u64,
    /// Voxel layers swept, and voxels tested in them.
    pub layers: u64,
    pub voxels: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Walker {
    pub params: Params,
    pub feet: [f64; 3],
    pub vel: [f64; 3],
    pub grounded: bool,
    /// Visual offset added to the eye height after a step-up or snap; decays to 0.
    pub eye_offset: f64,
    pub stats: Stats,
    prev_feet: [f64; 3],
    prev_offset: f64,
    accum: f64,
}

/// Voxel indices the half-open span `[lo, hi)` overlaps, with the [`EPS`] tolerance.
fn cells(lo: f64, hi: f64) -> std::ops::Range<i32> {
    (lo + EPS).floor() as i32..(hi - EPS).ceil() as i32
}

impl Walker {
    pub fn new(params: Params, feet: [f64; 3]) -> Self {
        Walker { params, feet, vel: [0.0; 3], grounded: false, eye_offset: 0.0, stats: Stats::default(), prev_feet: feet, prev_offset: 0.0, accum: 0.0 }
    }

    /// The box as `(min, max)` corners.
    pub fn aabb(&self) -> ([f64; 3], [f64; 3]) {
        let (w, f) = (self.params.half_width, self.feet);
        ([f[0] - w, f[1], f[2] - w], [f[0] + w, f[1] + self.params.height, f[2] + w])
    }

    /// The eye after the last whole substep.
    pub fn eye(&self) -> [f64; 3] {
        [self.feet[0], self.feet[1] + self.params.eye_height + self.eye_offset, self.feet[2]]
    }

    /// The eye interpolated into the unsimulated remainder of the frame, for smooth rendering.
    pub fn render_eye(&self) -> [f64; 3] {
        let t = (self.accum / self.params.substep_s).clamp(0.0, 1.0);
        let l = |a: f64, b: f64| a + (b - a) * t;
        let off = l(self.prev_offset, self.eye_offset);
        [l(self.prev_feet[0], self.feet[0]), l(self.prev_feet[1], self.feet[1]) + self.params.eye_height + off, l(self.prev_feet[2], self.feet[2])]
    }

    /// Whether the box overlaps any solid voxel. The invariant every substep keeps.
    pub fn overlaps<S: Solid>(&self, s: &S) -> bool {
        let (lo, hi) = self.aabb();
        cells(lo[1], hi[1]).any(|y| cells(lo[2], hi[2]).any(|z| cells(lo[0], hi[0]).any(|x| s.solid(x, y, z))))
    }

    /// Lifts the walker out of solid voxels, one voxel at a time, by at most `max_rise`. Returns
    /// whether it is free. Use when placing the walker (spawn, or leaving fly mode).
    pub fn resolve_start<S: Solid>(&mut self, s: &S, max_rise: i32) -> bool {
        for _ in 0..max_rise {
            if !self.overlaps(s) {
                break;
            }
            self.feet[1] = self.feet[1].floor() + 1.0;
        }
        self.prev_feet = self.feet;
        self.vel = [0.0; 3];
        !self.overlaps(s)
    }

    /// Whether the voxel layer `k` on `axis` holds a solid voxel inside the box's cross-section.
    fn layer_solid<S: Solid>(&mut self, s: &S, axis: usize, k: i32) -> bool {
        let (lo, hi) = self.aabb();
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        self.stats.layers += 1;
        for i in cells(lo[a], hi[a]) {
            for j in cells(lo[b], hi[b]) {
                self.stats.voxels += 1;
                let mut c = [0; 3];
                c[axis] = k;
                c[a] = i;
                c[b] = j;
                if s.solid(c[0], c[1], c[2]) {
                    return true;
                }
            }
        }
        false
    }

    /// Moves the box by `d` along `axis`, stopping flush against the first solid layer. Returns
    /// the signed distance moved and whether it was blocked.
    fn sweep<S: Solid>(&mut self, s: &S, axis: usize, d: f64) -> (f64, bool) {
        if d == 0.0 {
            return (0.0, false);
        }
        let start = self.feet[axis];
        let (lo, hi) = self.aabb();
        // Offsets of the box faces from the feet on this axis, exact, so a blocked box lands flush.
        let (off_lo, off_hi) = if axis == 1 { (0.0, self.params.height) } else { (-self.params.half_width, self.params.half_width) };
        if d > 0.0 {
            let (first, end) = ((hi[axis] - EPS).ceil() as i32, (hi[axis] + d - EPS).ceil() as i32);
            for k in first..end {
                if self.layer_solid(s, axis, k) {
                    self.feet[axis] = k as f64 - off_hi;
                    return (self.feet[axis] - start, true);
                }
            }
        } else {
            let (first, end) = ((lo[axis] + EPS).floor() as i32 - 1, (lo[axis] + d + EPS).floor() as i32);
            for k in (end..=first).rev() {
                if self.layer_solid(s, axis, k) {
                    self.feet[axis] = (k + 1) as f64 - off_lo;
                    return (self.feet[axis] - start, true);
                }
            }
        }
        self.feet[axis] = start + d;
        (d, false)
    }

    /// Advances by a frame of `dt` seconds: whole substeps, the remainder carried to the next frame.
    pub fn update<S: Solid>(&mut self, s: &S, intent: Intent, dt: f64) {
        self.accum += dt.clamp(0.0, self.params.max_frame_s);
        while self.accum >= self.params.substep_s {
            self.accum -= self.params.substep_s;
            self.substep(s, intent);
        }
    }

    /// One fixed substep.
    pub fn substep<S: Solid>(&mut self, s: &S, intent: Intent) {
        let p = self.params;
        let dt = p.substep_s;
        self.prev_feet = self.feet;
        self.prev_offset = self.eye_offset;
        self.stats.substeps += 1;
        let was_grounded = self.grounded;

        let len = (intent.dir[0] * intent.dir[0] + intent.dir[1] * intent.dir[1]).sqrt();
        let scale = if len > 1.0 { 1.0 / len } else { 1.0 };
        let speed = if intent.run { p.run_speed } else { p.walk_speed };
        self.vel[0] = intent.dir[0] * scale * speed;
        self.vel[2] = intent.dir[1] * scale * speed;
        let jumped = intent.jump && was_grounded;
        if jumped {
            self.vel[1] = p.jump_speed;
            self.grounded = false;
            self.stats.jumps += 1;
        }

        // Horizontal, with step-up.
        let (dx, dz) = (self.vel[0] * dt, self.vel[2] * dt);
        let start = self.feet;
        let (_, bx) = self.sweep(s, 0, dx);
        let (_, bz) = self.sweep(s, 2, dz);
        if was_grounded && !jumped && (bx || bz) && p.step_height > 0.0 {
            // Voxel tops are whole numbers, so try each whole level up to the step height, lowest
            // first: rising higher than needed could hit a low ceiling the lower level clears.
            let plain = self.feet;
            let progress = |f: [f64; 3]| (f[0] - start[0]).powi(2) + (f[2] - start[2]).powi(2);
            let mut stepped = None;
            let mut level = start[1].floor() + 1.0;
            while level <= start[1] + p.step_height + EPS {
                self.feet = start;
                let want = level - start[1];
                let (up, _) = self.sweep(s, 1, want);
                if up < want - EPS {
                    break; // no headroom for this level or any higher one
                }
                self.sweep(s, 0, dx);
                self.sweep(s, 2, dz);
                self.sweep(s, 1, -up);
                if progress(self.feet) > progress(plain) + 1e-12 {
                    stepped = Some(self.feet);
                    break;
                }
                level += 1.0;
            }
            match stepped {
                Some(f) => {
                    self.feet = f;
                    let rise = f[1] - start[1];
                    if rise > 0.0 {
                        self.stats.steps_up += 1;
                        self.eye_offset -= rise;
                    }
                }
                None => self.feet = plain,
            }
        }

        // Vertical.
        self.vel[1] = (self.vel[1] - p.gravity * dt).max(-p.max_fall_speed);
        let dy = self.vel[1] * dt;
        let (_, by) = self.sweep(s, 1, dy);
        self.grounded = by && dy < 0.0;
        if by {
            self.vel[1] = 0.0;
        }
        if !self.grounded && was_grounded && !jumped && self.vel[1] <= 0.0 {
            let y0 = self.feet[1];
            let (_, landed) = self.sweep(s, 1, -p.step_height);
            if landed {
                self.grounded = true;
                self.vel[1] = 0.0;
                self.eye_offset += y0 - self.feet[1];
                self.stats.snaps_down += 1;
            } else {
                self.feet[1] = y0;
            }
        }

        let lim = p.step_height;
        self.eye_offset = (self.eye_offset * (-p.eye_settle_per_s * dt).exp()).clamp(-lim, lim);
    }
}
