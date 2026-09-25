//! Phase 3A light transport (ADR-0005): units, the sun path, the physical atmosphere and the CPU
//! reference path tracer. Pure CPU; `gpu::reference` implements the same estimator on the GPU and is
//! checked against this crate.
//!
//! - [`sample`]: the shared RNG (PCG32) and direction samplers with their PDFs.
//! - [`sun`]: the unit of light (E_SUN = 1), the sun disk and [`sun::SunPath`] through the day.
//! - [`atmosphere`]: the Rayleigh + Mie + ozone atmosphere, transmittance and the Monte Carlo sky.
//! - [`emitters`] (4A): the emitter table and its alias sampler (ADR-0005 Amendment 3), and the rule
//!   that switches the lights ([`emitters::lights_on`]).
//! - [`exposure`] (4A): the metric exposure of a reference image.
//! - [`reference`]: the CPU path tracer over `world::reference::trace`.
//! - [`sky`] (3C): the real-time sky tables (Hillaire 2020), the oracle for `gpu/shaders/sky.slang`.
//! - [`sky_ref`] (S-020): the baked reference sky that corrects the sky-view table, and its file.

pub mod atmosphere;
pub mod emitters;
pub mod exposure;
pub mod reference;
pub mod sample;
pub mod sky;
pub mod sky_ref;
pub mod sun;

pub use atmosphere::{Atmosphere, SkyOptions};
pub use reference::{Lighting, Pinhole, Settings};
pub use sun::SunPath;
