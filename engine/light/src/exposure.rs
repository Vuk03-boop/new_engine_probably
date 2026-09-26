//! Phase 4A: the metric exposure of a reference image (Phase 4 proposal, decision 3).
//!
//! Comparisons on display values (FLIP) need an exposure. M3 took it from the sun and sky
//! irradiance, which goes to zero at night; M4 takes it from the reference image: the key 0.18 over
//! the image's log-average luminance. Pixels darker than 2⁻¹⁰ of the image mean are left out, so the
//! black night sky and noise-level pixels do not set it; by day no pixel is that dark and it is the
//! plain log-average. It scales with the image: k times the image gives 1/k times the exposure.

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

/// 4B: the time constant of the viewer's automatic exposure, seconds.
pub const ADAPT_SECONDS: f64 = 1.0;
/// 4B: the automatic exposure's bounds.
pub const MIN_EXPOSURE: f64 = 0.25;
pub const MAX_EXPOSURE: f64 = 65536.0;

/// 4B: the sums the viewer's meter (`gpu::compose`) takes over a shown image: ΣY over all pixels,
/// and Σ ln Y with a count over the pixels with Y > 0 and Y ≥ the floor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Meter {
    pub pixels: u64,
    pub sum_y: f64,
    pub sum_ln: f64,
    pub count: u64,
}

impl Meter {
    /// The host's oracle of the GPU meter, with the same rule.
    pub fn of(luminance: &[f64], floor: f64) -> Meter {
        let mut m = Meter { pixels: luminance.len() as u64, ..Meter::default() };
        for &y in luminance {
            m.sum_y += y;
            if y > 0.0 && y >= floor {
                m.sum_ln += y.ln();
                m.count += 1;
            }
        }
        m
    }

    /// The mean luminance, if any pixel was metered.
    pub fn mean(&self) -> Option<f64> {
        (self.pixels > 0).then(|| self.sum_y / self.pixels as f64)
    }

    /// The floor for the next reading: [`DARK_FRACTION`] of this reading's mean (0 without one).
    pub fn next_floor(&self) -> f64 {
        self.mean().filter(|m| m.is_finite() && *m > 0.0).map_or(0.0, |m| m * DARK_FRACTION)
    }

    /// [`KEY`] over the log-average of the counted pixels; `None` without light.
    pub fn target(&self) -> Option<f64> {
        (self.count > 0).then(|| KEY / (self.sum_ln / self.count as f64).exp()).filter(|t| t.is_finite() && *t > 0.0)
    }
}

/// 4B: one step of the automatic exposure: in log₂, a fraction 1 − e^(−dt / [`ADAPT_SECONDS`]) of the
/// way to `target`, then bounded to [[`MIN_EXPOSURE`], [`MAX_EXPOSURE`]]. No target keeps it.
pub fn adapt(exposure: f64, target: Option<f64>, dt: f64) -> f64 {
    let Some(t) = target else { return exposure.clamp(MIN_EXPOSURE, MAX_EXPOSURE) };
    let (ev, ev_t) = (exposure.log2(), t.log2());
    let k = 1.0 - (-dt.max(0.0) / ADAPT_SECONDS).exp();
    (ev + (ev_t - ev) * k).exp2().clamp(MIN_EXPOSURE, MAX_EXPOSURE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C8 (4B): the adaptation and the target from the meter's sums.
    #[test]
    fn automatic_exposure_adapts_and_matches_the_metric() {
        let near = |a: f64, b: f64| ((a - b) / b).abs() <= 1e-12;
        // After one time constant, 1 − 1/e of the way in log₂.
        let (e0, t) = (4.0f64, 1024.0f64);
        let e1 = adapt(e0, Some(t), ADAPT_SECONDS);
        let frac = (e1.log2() - e0.log2()) / (t.log2() - e0.log2());
        assert!((frac - (1.0 - (-1.0f64).exp())).abs() <= 1e-12, "{frac}");
        // Frame-rate independent: two half steps equal one step.
        for dt in [1.0 / 144.0, 1.0 / 60.0, 0.1, 0.7] {
            assert!(near(adapt(adapt(e0, Some(t), dt / 2.0), Some(t), dt / 2.0), adapt(e0, Some(t), dt)), "dt {dt}");
        }
        // Bounded; no reading keeps it.
        assert_eq!(adapt(100.0, Some(1e9), 100.0), MAX_EXPOSURE);
        assert_eq!(adapt(100.0, Some(1e-9), 100.0), MIN_EXPOSURE);
        assert!(near(adapt(37.0, None, 0.5), 37.0));
        assert!(near(adapt(37.0, Some(1000.0), 0.0), 37.0));
        // The target from the sums is the metric exposure when the floor is the image's own.
        let mut img: Vec<f64> = (0..500).map(|i| 1e-4 * (1.0 + (i % 13) as f64) * if i % 7 == 0 { 40.0 } else { 1.0 }).collect();
        img.extend(std::iter::repeat_n(0.0, 300));
        img.extend([1e-12, 3e-13]);
        let own = Meter::of(&img, 0.0).next_floor();
        let m = Meter::of(&img, own);
        assert!(near(m.target().unwrap(), metric_exposure(&img).unwrap()));
        assert_eq!(Meter::of(&[0.0; 16], 0.0).target(), None);
    }

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
}
