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
}
