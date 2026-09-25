//! Phase 4A: the metric exposure of a reference image (Phase 4 proposal, decision 3).
//!
//! Comparisons on display values (FLIP) need an exposure. M3 took it from the sun and sky
//! irradiance, which goes to zero at night; M4 takes it from the reference image: the key 0.18 over
//! the image's log-average luminance. Pixels darker than 2⁻¹⁰ of the image mean are left out, so the
//! black night sky and noise-level pixels do not set it; by day no pixel is that dark and it is the
//! plain log-average. It scales with the image: k times the image gives 1/k times the exposure.
//!
//! 4B: the viewer's automatic exposure uses the same rule on the shown image: the GPU sums the
//! displayed luminance and its log over the pixels at least [`DARK_FRACTION`] of the previous frame's
//! mean ([`LogSums`], mirrored on the host by [`log_sums`]), [`exposure_from_sums`] turns them into a
//! target, and [`Adaptation`] follows it with a time constant of [`ADAPT_SECONDS`], bounded.

/// The display key: the log-average luminance maps to this.
pub const KEY: f64 = 0.18;
/// Pixels below this fraction of the image's mean luminance are left out of the log-average.
pub const DARK_FRACTION: f64 = 1.0 / 1024.0;

/// [`KEY`] over the log-average luminance of the pixels at least [`DARK_FRACTION`] of the mean.
/// `None` when the image has no light (its mean is not finite and positive).
pub fn metric_exposure(luminance: &[f64]) -> Option<f64> {
    let mean = luminance.iter().sum::<f64>() / luminance.len() as f64;
    if !(mean.is_finite() && mean > 0.0) {
        return None;
    }
    let floor = mean * DARK_FRACTION;
    // At least one pixel is at or above the mean, so the set is never empty.
    let (sum, n) = luminance.iter().filter(|&&y| y >= floor).fold((0.0, 0usize), |(s, n), &y| (s + y.ln(), n + 1));
    Some(KEY / (sum / n as f64).exp())
}

/// 4B: the viewer's adaptation time constant, seconds.
pub const ADAPT_SECONDS: f64 = 1.0;
/// 4B: the adapted exposure stays within these (day about 1-100, night about 10⁴).
pub const MIN_EXPOSURE: f64 = 0.25;
pub const MAX_EXPOSURE: f64 = 65536.0;

/// The sums behind an exposure: of the luminance and its count, and of ln Y and its count over the
/// pixels with Y ≥ the floor and Y > 0 (`gpu::exposure` writes the same, in f32).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogSums {
    pub sum: f64,
    pub count: f64,
    pub sum_log: f64,
    pub lit: f64,
}

impl LogSums {
    /// The mean luminance (the next frame's floor is this times [`DARK_FRACTION`]).
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0.0).then(|| self.sum / self.count)
    }
}

/// [`LogSums`] of `luminance` with `floor` (the host mirror of the GPU pass).
pub fn log_sums(luminance: &[f64], floor: f64) -> LogSums {
    let mut s = LogSums::default();
    for &y in luminance {
        s.sum += y;
        s.count += 1.0;
        if y >= floor && y > 0.0 {
            s.sum_log += y.ln();
            s.lit += 1.0;
        }
    }
    s
}

/// [`KEY`] over the log-average of the lit pixels; `None` without any.
pub fn exposure_from_sums(s: &LogSums) -> Option<f64> {
    (s.lit > 0.0 && s.sum_log.is_finite()).then(|| KEY / (s.sum_log / s.lit).exp())
}

/// The viewer's adapted exposure: the first target is taken at once; later the log exposure moves
/// toward each target with time constant [`ADAPT_SECONDS`]; it stays within [[`MIN_EXPOSURE`],
/// [`MAX_EXPOSURE`]].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Adaptation {
    log2: Option<f64>,
}

impl Adaptation {
    /// Moves toward `target` over `dt` seconds; returns the adapted exposure.
    pub fn update(&mut self, target: f64, dt: f64) -> f64 {
        let t = target.clamp(MIN_EXPOSURE, MAX_EXPOSURE).log2();
        let next = match self.log2 {
            None => t,
            Some(l) => l + (t - l) * (1.0 - (-dt.max(0.0) / ADAPT_SECONDS).exp()),
        };
        let next = next.clamp(MIN_EXPOSURE.log2(), MAX_EXPOSURE.log2());
        self.log2 = Some(next);
        next.exp2()
    }

    /// The adapted exposure times 2^`stops` (the viewer's − and =); `None` before the first target.
    pub fn exposure(&self, stops: f64) -> Option<f64> {
        self.log2.map(|l| (l + stops).exp2())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        ((a - b) / b).abs() <= 1e-12
    }

    /// C7 (4A part 2).
    #[test]
    fn exposure_follows_the_lit_pixels() {
        assert!(close(metric_exposure(&[0.25; 16]).unwrap(), 0.18 / 0.25));
        // A day-like image: nothing below 2⁻¹⁰ of the mean, so the plain log-average.
        let day: Vec<f64> = (0..64).map(|i| 0.01 + 0.03 * (i % 7) as f64 + 0.2 * (i % 3) as f64).collect();
        let plain = (day.iter().map(|y| y.ln()).sum::<f64>() / day.len() as f64).exp();
        assert!(close(metric_exposure(&day).unwrap(), 0.18 / plain));
        // Scaling the image by k divides the exposure by k.
        for k in [1e-6, 3.7e-3, 42.0] {
            let scaled: Vec<f64> = day.iter().map(|y| y * k).collect();
            assert!(close(metric_exposure(&scaled).unwrap(), metric_exposure(&day).unwrap() / k), "k = {k}");
        }
        // Black pixels (a night sky) do not change it: no pixel lies between the old and new floors.
        let mut night = day.iter().map(|y| y * 1e-4).collect::<Vec<f64>>();
        let lit = metric_exposure(&night).unwrap();
        night.extend(std::iter::repeat_n(0.0, 40));
        assert!(close(metric_exposure(&night).unwrap(), lit));
        // Nor do noise-level pixels below the floor; a lit one does.
        let mut noisy = night.clone();
        noisy.extend([1e-13, 3e-14]);
        assert!(close(metric_exposure(&noisy).unwrap(), lit));
        noisy.push(1e-3);
        assert!(!close(metric_exposure(&noisy).unwrap(), lit));
        assert_eq!(metric_exposure(&[0.0; 8]), None);
        assert_eq!(metric_exposure(&[]), None);
    }

    /// 4B C3: the adaptation's first target, time constant, bounds and stops; and the sums the GPU
    /// writes give `metric_exposure` when the floor is the image's own mean x 2⁻¹⁰.
    #[test]
    fn adaptation_follows_the_target() {
        let mut a = Adaptation::default();
        assert_eq!(a.exposure(0.0), None);
        assert!(close(a.update(100.0, 0.5), 100.0), "the first target is taken at once");
        for _ in 0..60 {
            a.update(6400.0, 1.0 / 60.0);
        }
        let closed = (a.exposure(0.0).unwrap().log2() - 100f64.log2()) / (6400f64.log2() - 100f64.log2());
        assert!((closed - (1.0 - (-1f64).exp())).abs() <= 0.01 * (1.0 - (-1f64).exp()), "closed {closed}");
        assert!(close(a.exposure(2.0).unwrap(), 4.0 * a.exposure(0.0).unwrap()));
        for _ in 0..1000 {
            a.update(1e9, 0.1);
        }
        assert!(close(a.exposure(0.0).unwrap(), MAX_EXPOSURE));
        for _ in 0..1000 {
            a.update(1e-9, 0.1);
        }
        assert!(close(a.exposure(0.0).unwrap(), MIN_EXPOSURE));

        let night: Vec<f64> = (0..500).map(|i| if i % 5 == 0 { 0.0 } else { 1e-5 * (1.0 + (i % 17) as f64) * if i % 3 == 0 { 1e-6 } else { 1.0 } }).collect();
        let mean = night.iter().sum::<f64>() / night.len() as f64;
        let s = log_sums(&night, mean * DARK_FRACTION);
        assert_eq!(s.count, 500.0);
        assert!(close(exposure_from_sums(&s).unwrap(), metric_exposure(&night).unwrap()));
        assert!(s.lit < 400.0, "the noise-level pixels are left out");
        assert_eq!(exposure_from_sums(&log_sums(&[0.0; 4], 0.0)), None);
    }
}
