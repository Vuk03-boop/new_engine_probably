//! The sun: units and its path through the day (ADR-0005).
//!
//! World axes: +X east, +Y up, −Z north. Solar irradiance at the top of the atmosphere is 1 per
//! channel; the disk is uniform with angular radius [`SUN_ANGULAR_RADIUS`].

use std::f64::consts::PI;

use crate::sample::V3;

/// Top-of-atmosphere solar irradiance per channel, perpendicular to the sun (the unit of light).
pub const E_SUN: f64 = 1.0;
/// Angular radius of the sun disk, radians (0.2664°).
pub const SUN_ANGULAR_RADIUS: f64 = 0.2664 * PI / 180.0;

pub fn sun_cos_max() -> f64 {
    SUN_ANGULAR_RADIUS.cos()
}

/// Solid angle of the sun disk.
pub fn sun_solid_angle() -> f64 {
    2.0 * PI * (1.0 - sun_cos_max())
}

/// Solar declination and observer latitude; the hour angle comes from the time of day.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SunPath {
    pub latitude_deg: f64,
    pub declination_deg: f64,
}

impl Default for SunPath {
    fn default() -> SunPath {
        SunPath { latitude_deg: 45.0, declination_deg: 0.0 }
    }
}

impl SunPath {
    /// Unit vector toward the sun at `hour` (local solar time, 12 = noon).
    pub fn direction(&self, hour: f64) -> V3 {
        let (phi, delta) = (self.latitude_deg.to_radians(), self.declination_deg.to_radians());
        let h = ((hour - 12.0) * 15.0).to_radians();
        let up = phi.sin() * delta.sin() + phi.cos() * delta.cos() * h.cos();
        let east = -delta.cos() * h.sin();
        let north = phi.cos() * delta.sin() - phi.sin() * delta.cos() * h.cos();
        [east, up, -north]
    }
}

/// Elevation above the horizon, degrees.
pub fn elevation_deg(dir: V3) -> f64 {
    dir[1].clamp(-1.0, 1.0).asin().to_degrees()
}

/// The reference times of ADR-0005: (name, hour).
pub const REFERENCE_TIMES: [(&str, f64); 5] = [("dawn", 6.25), ("morning", 8.0), ("midday", 12.0), ("dusk", 17.75), ("twilight", 18.25)];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::dot;

    #[test]
    fn equinox_path_at_45_degrees() {
        let p = SunPath::default();
        let noon = p.direction(12.0);
        assert!((elevation_deg(noon) - 45.0).abs() < 1e-9);
        assert!(noon[2] > 0.0 && noon[0].abs() < 1e-12, "noon sun is due south (+Z): {noon:?}");
        let morning = p.direction(8.0);
        assert!(morning[0] > 0.0, "morning sun is in the east");
        assert!(p.direction(16.0)[0] < 0.0, "afternoon sun is in the west");
        assert!(elevation_deg(p.direction(6.0)).abs() < 1e-9, "equinox sunrise at 6 h");
        for (name, hour) in REFERENCE_TIMES {
            let d = p.direction(hour);
            assert!((dot(d, d) - 1.0).abs() < 1e-12, "{name}");
        }
        assert!((elevation_deg(p.direction(6.25)) - 2.65).abs() < 0.01);
        assert!((elevation_deg(p.direction(18.25)) + 2.65).abs() < 0.01);
    }

    #[test]
    fn sun_radiance_times_solid_angle_is_the_irradiance() {
        let l = E_SUN / sun_solid_angle();
        assert!((l * sun_solid_angle() - E_SUN).abs() < 1e-12);
        assert!(sun_solid_angle() > 6.7e-5 && sun_solid_angle() < 6.9e-5);
    }
}
