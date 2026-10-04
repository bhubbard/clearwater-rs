# Clearwater (Rust) 🌊

[![Crates.io](https://img.shields.io/crates/v/clearwater-rs.svg)](https://crates.io/crates/clearwater-rs)
[![Documentation](https://docs.rs/clearwater-rs/badge.svg)](https://docs.rs/clearwater-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance, photoreal shallow water physics and rendering engine in Rust. Ported and extended from [Aurélien's Clearwater](https://github.com/Aureliengmz/clearwater).

---

## Features

- **Jerry Tessendorf 2D FFT Ocean Wave Spectrum**:
  - Phillips / JONSWAP-style directional wave dispersion relation with shallow water tanh factor $\omega(k) = \sqrt{(g k + \sigma k^3) \tanh(k d)}$.
  - Continuous physical slope normalization targeting realistic sea states (RMS slope $\approx 0.078$).
  - Choppy horizontal displacement vectors $(\Delta x, \eta, \Delta z)$ and gradient normal mapping.
- **Evan Wallace Refracted-Grid Caustics**:
  - Direct photon density accumulation on the seabed with flux conservation.
  - Multi-wavelength chromatic dispersion ($IOR_R = 1.3315$, $IOR_G = 1.3335$, $IOR_B = 1.3365$) producing vibrant spectral fringing along caustic bands.
  - Multi-touch ripple curvature modulation using the discrete surface Laplacian $\nabla^2 h$.
- **2D Wave Equation Interactive Ripples**:
  - Localized $256 \times 256$ dynamic simulation grid tracking viewer attention across open water.
  - Smooth boundary dampening with cosine droplet perturbation impulses.
- **Marc Olano & Dan Baker LEAN Mapping**:
  - Slope covariance matrix computation and Beckmann microfacet variance widening to prevent aliasing and sparkle flicker on distant wave horizons.
- **Physical Shallow Water Optics**:
  - Unpolarized dielectric Fresnel reflectance and Snell's law refraction.
  - Beer-Lambert volumetric absorption ($\sigma_a = [0.40, 0.074, 0.088]\ \text{m}^{-1}$) and scattering ($\sigma_s = [0.028, 0.052, 0.068]\ \text{m}^{-1}$), producing crystalline turquoise water coloration.
  - Henyey-Greenstein forward phase function for suspended particulate volume.
- **Procedural Seabed Synthesis**:
  - Dart-throwing packing of super-elliptical stones on a 2-torus.
  - 12-color mineral palette, quartz veins, ambient occlusion baking, directional sunlight, and contact shadows.
- **Standalone Single-File WebGL2 Exporter**:
  - Generates zero-dependency standalone HTML files with embedded shaders and simulation kernels.
- **Interactive Preview Server**:
  - Built-in `axum` / `tokio` HTTP server with live REST API for heightfield and caustics inspection.

---

## Installation

Add `clearwater-rs` to your `Cargo.toml`:

```toml
[dependencies]
clearwater-rs = "0.1.0"
```

Or install the CLI tool:

```bash
cargo install clearwater-rs
```

---

## CLI Usage

### 1. Launch the Interactive Local Web Server

```bash
clearwater serve --port 8080 --open
```

Visit `http://localhost:8080` to interact with the real-time simulation, look around, and tap to create water ripples.

### 2. Export Standalone Self-Contained HTML

Generate a single self-contained `.html` file that runs in any browser with WebGL2 (no external assets or internet connection required):

```bash
clearwater export --output clearwater.html
```

### 3. Procedurally Synthesize Seamless Pebble Bed Textures

Generate a seamless 1024x1024 mineral stone texture matching the seabed algorithm:

```bash
clearwater pebbles --output pebbles.png --size 1024 --seed 11
```

### 4. Evaluate Wave Simulation & Caustics to Images

```bash
clearwater sim --time 3.5 --out-dir output/
```

### 5. Benchmark Performance

```bash
clearwater bench --iterations 200
```

---

## Rust Library API Example

```rust
use clearwater::{
    H0Spectrum, OceanSpectrumConfig, OceanFftSolver,
    CausticEngine, CausticConfig,
    RippleSimulation, RippleDrop,
};
use glam::Vec3;

fn main() {
    // 1. Configure and initialize the Tessendorf ocean spectrum
    let config = OceanSpectrumConfig {
        grid_size: 256,
        patch_size: 4.6,
        depth: 1.6,
        target_slope: 0.078,
        ..Default::default()
    };
    let spectrum = H0Spectrum::new(config);

    // 2. Evaluate spectrum at t = 2.5 seconds
    let frame = spectrum.evaluate(2.5);

    // 3. Solve 2D IFFT to obtain spatial heights, slopes, and normals
    let mut solver = OceanFftSolver::new(256);
    let wave = solver.solve(&frame, 4.6);

    println!("Wave height at (0, 0): {:.3} m", wave.sample_height_bilinear(0.0, 0.0));
    println!("Average slope variance: {:.5}", wave.slope_variance);

    // 4. Render refracted-grid caustics with chromatic dispersion
    let caustic_engine = CausticEngine::new(CausticConfig::default());
    let sun_dir = Vec3::new(0.4, 0.8, -0.4).normalize();
    let caustics = caustic_engine.render(&wave, sun_dir);

    println!("Caustic resolution: {}x{}", caustics.resolution, caustics.resolution);
}
```

---

## References

- Jerry Tessendorf, *Simulating Ocean Water* (2001)
- Evan Wallace, *WebGL Water*
- Marc Olano & Dan Baker, *LEAN Mapping: Linear Efficient Antialiased Normal Mapping*, I3D (2010)
- Inigo Quilez, *Texture Repetition and Voronoi Metrics*

---

## Credits & License

- Original WebGL Clearwater engine by [Aurélien G.](https://x.com/Aurelien_Gz) at [Lumaris](https://lumaris.works).
- Rust architecture and CLI engine by [Brandon Hubbard](https://github.com/bhubbard).
- Licensed under the [MIT License](LICENSE).
