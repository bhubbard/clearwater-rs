use glam::Vec3;
use crate::optics::{refract, IOR_BLUE, IOR_DEFAULT, IOR_GREEN, IOR_RED};
use crate::fft::WaveField2D;

/// Configuration for the refracted-grid caustics solver.
#[derive(Debug, Clone)]
pub struct CausticConfig {
    /// Number of grid rays along each axis (e.g. 256 for a 256x256 ray grid).
    pub ray_grid_size: usize,
    /// Texture map resolution for photon accumulation (e.g. 1024x1024).
    pub caustic_res: usize,
    /// Patch size in meters (e.g. 4.6 m).
    pub patch_size: f32,
    /// Mean seabed depth in meters (e.g. 1.6 m).
    pub depth: f32,
}

impl Default for CausticConfig {
    fn default() -> Self {
        Self {
            ray_grid_size: 256,
            caustic_res: 1024,
            patch_size: 4.6,
            depth: 1.6,
        }
    }
}

/// Evaluated caustic illumination map holding RGB photon intensity values.
#[derive(Debug, Clone)]
pub struct CausticMap {
    pub resolution: usize,
    /// RGB floating-point intensities for each texel in `[0..1, 0..1]` floor UV space.
    pub rgb: Vec<Vec3>,
    /// Bulk flat-refraction shift offset vector `(dx, dz)` in meters.
    pub shift: [f32; 2],
}

impl CausticMap {
    /// Samples RGB caustic intensity at continuous physical seabed coordinates `(x, z)`.
    pub fn sample_bilinear(&self, x: f32, z: f32, patch_size: f32) -> Vec3 {
        let u = ((x - self.shift[0]) / patch_size).rem_euclid(1.0);
        let v = ((z - self.shift[1]) / patch_size).rem_euclid(1.0);

        let res = self.resolution as f32;
        let px = u * res;
        let py = v * res;

        let x0 = (px.floor() as usize) % self.resolution;
        let y0 = (py.floor() as usize) % self.resolution;
        let x1 = (x0 + 1) % self.resolution;
        let y1 = (y0 + 1) % self.resolution;

        let fx = px.fract();
        let fy = py.fract();

        let c00 = self.rgb[y0 * self.resolution + x0];
        let c10 = self.rgb[y0 * self.resolution + x1];
        let c01 = self.rgb[y1 * self.resolution + x0];
        let c11 = self.rgb[y1 * self.resolution + x1];

        let top = c00.lerp(c10, fx);
        let bot = c01.lerp(c11, fx);
        top.lerp(bot, fy)
    }
}

/// Evan Wallace refracted-grid caustics engine.
pub struct CausticEngine {
    pub config: CausticConfig,
}

impl CausticEngine {
    pub fn new(config: CausticConfig) -> Self {
        Self { config }
    }

    /// Computes the complete caustic photon map by ray-tracing the perturbed wave surface down to the seabed.
    pub fn render(&self, wave: &WaveField2D, sun_dir: Vec3) -> CausticMap {
        let g = self.config.ray_grid_size;
        let c = self.config.caustic_res;
        let l = self.config.patch_size;
        let depth = self.config.depth;

        // 1. Calculate flat-surface refraction shift vector using middle wavelength (green)
        let sy = sun_dir.y.clamp(0.01, 1.0);
        let sin_i = (1.0 - sy * sy).max(0.0).sqrt();
        let sin_t = sin_i / IOR_DEFAULT;
        let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
        let tan_t = sin_t / cos_t.max(1e-4);
        let hd = sun_dir.x.hypot(sun_dir.z).max(1e-4);
        let shift = [
            -sun_dir.x / hd * depth * tan_t,
            -sun_dir.z / hd * depth * tan_t,
        ];

        let mut caustic_rgb = vec![Vec3::ZERO; c * c];
        let iors = [IOR_RED, IOR_GREEN, IOR_BLUE];

        // 2. Trace refracted ray grid for each wavelength
        let cell_uv_size = 1.0 / (g as f32);
        let norm_flux = (c as f32 / l) * (c as f32 / l) * (cell_uv_size * cell_uv_size) * 1.8;

        for (channel, &ior) in iors.iter().enumerate() {
            for j in 0..g {
                let v0 = (j as f32) / (g as f32);
                let z0 = v0 * l;

                for i in 0..g {
                    let u0 = (i as f32) / (g as f32);
                    let x0 = u0 * l;

                    let h0 = wave.sample_height_bilinear(x0, z0);
                    let n0 = wave.sample_normal(x0, z0);

                    // Refract incident sunlight ray through water surface
                    if let Some(r) = refract(-sun_dir, n0, 1.0 / ior) {
                        if r.y < -1e-4 {
                            // Intersect with seabed plane at y = -depth
                            let dist = (-depth - h0) / r.y;
                            let floor_x = x0 + r.x * dist;
                            let floor_z = z0 + r.z * dist;

                            // Map floor coordinates relative to shift offset
                            let target_u = ((floor_x - shift[0]) / l).rem_euclid(1.0);
                            let target_v = ((floor_z - shift[1]) / l).rem_euclid(1.0);

                            let tx = ((target_u * (c as f32)) as usize) % c;
                            let ty = ((target_v * (c as f32)) as usize) % c;
                            let idx = ty * c + tx;

                            // Accumulate energy into color channel
                            match channel {
                                0 => caustic_rgb[idx].x += norm_flux,
                                1 => caustic_rgb[idx].y += norm_flux,
                                2 => caustic_rgb[idx].z += norm_flux,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // Clamp intensity peaks and normalize
        for pixel in caustic_rgb.iter_mut() {
            pixel.x = pixel.x.min(40.0);
            pixel.y = pixel.y.min(40.0);
            pixel.z = pixel.z.min(40.0);
        }

        CausticMap {
            resolution: c,
            rgb: caustic_rgb,
            shift,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::OceanFftSolver;
    use crate::spectrum::{H0Spectrum, OceanSpectrumConfig};

    #[test]
    fn test_caustics_generation() {
        let config = OceanSpectrumConfig {
            grid_size: 32,
            patch_size: 4.0,
            ..Default::default()
        };
        let spectrum = H0Spectrum::new(config);
        let frame = spectrum.evaluate(0.0);
        let mut solver = OceanFftSolver::new(32);
        let wave = solver.solve(&frame, 4.0);

        let caustic_cfg = CausticConfig {
            ray_grid_size: 32,
            caustic_res: 64,
            patch_size: 4.0,
            depth: 1.5,
        };
        let engine = CausticEngine::new(caustic_cfg);
        let sun = Vec3::new(0.5, 0.8, -0.3).normalize();
        let map = engine.render(&wave, sun);

        assert_eq!(map.rgb.len(), 64 * 64);
        let sample = map.sample_bilinear(2.0, 2.0, 4.0);
        assert!(sample.x >= 0.0);
    }
}
