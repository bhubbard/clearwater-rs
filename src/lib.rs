//! # Clearwater
//!
//! Real-time, photoreal shallow water physics and rendering engine in Rust.
//!
//! Featuring:
//! - **Jerry Tessendorf 2D FFT Ocean Wave Spectrum**: Phillips / JONSWAP-style dispersion relation with shallow depth tanh factor.
//! - **Evan Wallace Refracted-Grid Caustics**: Direct photon accumulation on the seabed with multi-wavelength chromatic dispersion.
//! - **2D Wave Equation Interactive Ripples**: Dynamic moving window tracking user interaction with velocity and height damping.
//! - **Marc Olano & Dan Baker LEAN Mapping**: Slope covariance matrices and variance widening for anti-aliased distant sun glints.
//! - **Procedural Seabed Synthesis**: Dart-throwing packing on a 2-torus, mineral color palette, super-elliptical stone domes, quartz veins, and ambient occlusion.
//! - **Physical Water Optics**: Unpolarized dielectric Fresnel reflectance, Snell's law refraction, and Beer-Lambert spectral absorption.
//! - **Standalone HTML/WebGL2 Exporter & Local Preview Server**: Interactive web server and self-contained zero-dependency HTML generator.

pub mod spectrum;
pub mod fft;
pub mod ripples;
pub mod caustics;
pub mod optics;
pub mod seabed;
pub mod lean;
pub mod export;
pub mod server;

pub use spectrum::{H0Spectrum, OceanSpectrumConfig, SpectrumFrame};
pub use fft::{OceanFftSolver, WaveField2D};
pub use ripples::{RippleDrop, RippleSimulation};
pub use caustics::{CausticConfig, CausticEngine, CausticMap};
pub use optics::{fresnel, refract, reflect, beer_lambert, sky_radiance};
pub use seabed::{PebbleBedGenerator, floor_depth};
pub use lean::LeanCovariance;
pub use export::HtmlExporter;
