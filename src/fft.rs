use std::sync::Arc;
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use glam::Vec3;
use crate::spectrum::SpectrumFrame;

/// 2D Inverse FFT evaluator for ocean surface fields.
pub struct OceanFftSolver {
    grid_size: usize,
    fft_inverse_row: Arc<dyn Fft<f32>>,
    scratch_row: Vec<Complex32>,
}

impl OceanFftSolver {
    /// Creates a new solver with pre-planned FFT execution kernels.
    pub fn new(grid_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft_inverse_row = planner.plan_fft_inverse(grid_size);
        let scratch_len = fft_inverse_row.get_inplace_scratch_len();

        Self {
            grid_size,
            fft_inverse_row,
            scratch_row: vec![Complex32::default(); scratch_len],
        }
    }

    /// Performs an in-place 2D Inverse FFT on an $N \times N$ buffer (row-major).
    pub fn ifft2_inplace(&mut self, buffer: &mut [Complex32]) {
        let n = self.grid_size;
        assert_eq!(buffer.len(), n * n);

        // 1. Transform each row with 1D inverse FFT
        for y in 0..n {
            let row = &mut buffer[y * n..(y + 1) * n];
            self.fft_inverse_row.process_with_scratch(row, &mut self.scratch_row);
        }

        // 2. Transpose buffer
        for y in 0..n {
            for x in (y + 1)..n {
                buffer.swap(y * n + x, x * n + y);
            }
        }

        // 3. Transform each column (now row) with 1D inverse FFT
        for y in 0..n {
            let row = &mut buffer[y * n..(y + 1) * n];
            self.fft_inverse_row.process_with_scratch(row, &mut self.scratch_row);
        }

        // 4. Transpose back
        for y in 0..n {
            for x in (y + 1)..n {
                buffer.swap(y * n + x, x * n + y);
            }
        }

        // 5. Standard scale factor 1.0 (or normalized by 1 / (N*N) depending on convention)
        // Note: in Tessendorf waves, the random amplitude already carries 1/N factors,
        // but rustfft unnormalized inverse adds a factor of N in each dimension (N^2 total).
        // We scale by 1 / (N * N) so spatial domain values reflect physical heights in meters.
        let scale = 1.0 / ((n * n) as f32);
        for val in buffer.iter_mut() {
            *val *= scale;
        }
    }

    /// Evaluates the complete physical spatial wave surface from a frequency-domain `SpectrumFrame`.
    pub fn solve(&mut self, frame: &SpectrumFrame, patch_size: f32) -> WaveField2D {
        let n = self.grid_size;

        let mut h_buf = frame.height_k.clone();
        let mut sx_buf = frame.slope_x_k.clone();
        let mut sz_buf = frame.slope_z_k.clone();
        let mut dx_buf = frame.disp_x_k.clone();
        let mut dz_buf = frame.disp_z_k.clone();

        self.ifft2_inplace(&mut h_buf);
        self.ifft2_inplace(&mut sx_buf);
        self.ifft2_inplace(&mut sz_buf);
        self.ifft2_inplace(&mut dx_buf);
        self.ifft2_inplace(&mut dz_buf);

        let mut height = Vec::with_capacity(n * n);
        let mut slope_x = Vec::with_capacity(n * n);
        let mut slope_z = Vec::with_capacity(n * n);
        let mut normals = Vec::with_capacity(n * n);
        let mut disp_x = Vec::with_capacity(n * n);
        let mut disp_z = Vec::with_capacity(n * n);

        let mut slope_sq_sum = 0.0f32;

        for i in 0..(n * n) {
            let h = h_buf[i].re;
            let sx = sx_buf[i].re;
            let sz = sz_buf[i].re;
            let dx = dx_buf[i].re;
            let dz = dz_buf[i].re;

            // Surface normal from gradients: n = normalize(-dh/dx, 1, -dh/dz)
            let n_vec = Vec3::new(-sx, 1.0, -sz).normalize();

            height.push(h);
            slope_x.push(sx);
            slope_z.push(sz);
            normals.push(n_vec);
            disp_x.push(dx);
            disp_z.push(dz);

            slope_sq_sum += sx * sx + sz * sz;
        }

        let slope_variance = slope_sq_sum / ((n * n) as f32);

        WaveField2D {
            grid_size: n,
            patch_size,
            time: frame.time,
            height,
            slope_x,
            slope_z,
            normals,
            disp_x,
            disp_z,
            slope_variance,
        }
    }
}

/// Evaluated 2D spatial ocean wave field at time `t`.
#[derive(Debug, Clone)]
pub struct WaveField2D {
    pub grid_size: usize,
    pub patch_size: f32,
    pub time: f32,
    /// Vertical height displacement $\eta(x, z)$ in meters.
    pub height: Vec<f32>,
    /// Horizontal slope gradient $\partial h / \partial x$.
    pub slope_x: Vec<f32>,
    /// Horizontal slope gradient $\partial h / \partial z$.
    pub slope_z: Vec<f32>,
    /// Normalized surface normal vectors.
    pub normals: Vec<Vec3>,
    /// Choppy horizontal displacement in X (meters).
    pub disp_x: Vec<f32>,
    /// Choppy horizontal displacement in Z (meters).
    pub disp_z: Vec<f32>,
    /// Average slope variance $\sigma^2$ for LEAN anisotropic reflection widening.
    pub slope_variance: f32,
}

impl WaveField2D {
    /// Samples wave height at continuous physical coordinates `(x, z)` using bilinear filtering.
    pub fn sample_height_bilinear(&self, x: f32, z: f32) -> f32 {
        let n = self.grid_size as f32;
        let u = (x / self.patch_size).rem_euclid(1.0) * n;
        let v = (z / self.patch_size).rem_euclid(1.0) * n;

        let x0 = (u.floor() as usize) % self.grid_size;
        let y0 = (v.floor() as usize) % self.grid_size;
        let x1 = (x0 + 1) % self.grid_size;
        let y1 = (y0 + 1) % self.grid_size;

        let fx = u.fract();
        let fy = v.fract();

        let h00 = self.height[y0 * self.grid_size + x0];
        let h10 = self.height[y0 * self.grid_size + x1];
        let h01 = self.height[y1 * self.grid_size + x0];
        let h11 = self.height[y1 * self.grid_size + x1];

        let top = h00 * (1.0 - fx) + h10 * fx;
        let bot = h01 * (1.0 - fx) + h11 * fx;
        top * (1.0 - fy) + bot * fy
    }

    /// Samples wave normal vector at continuous physical coordinates `(x, z)`.
    pub fn sample_normal(&self, x: f32, z: f32) -> Vec3 {
        let n = self.grid_size as f32;
        let u = (x / self.patch_size).rem_euclid(1.0) * n;
        let v = (z / self.patch_size).rem_euclid(1.0) * n;

        let x0 = (u.floor() as usize) % self.grid_size;
        let y0 = (v.floor() as usize) % self.grid_size;
        let x1 = (x0 + 1) % self.grid_size;
        let y1 = (y0 + 1) % self.grid_size;

        let fx = u.fract();
        let fy = v.fract();

        let sx00 = self.slope_x[y0 * self.grid_size + x0];
        let sx10 = self.slope_x[y0 * self.grid_size + x1];
        let sx01 = self.slope_x[y1 * self.grid_size + x0];
        let sx11 = self.slope_x[y1 * self.grid_size + x1];
        let sx = (sx00 * (1.0 - fx) + sx10 * fx) * (1.0 - fy) + (sx01 * (1.0 - fx) + sx11 * fx) * fy;

        let sz00 = self.slope_z[y0 * self.grid_size + x0];
        let sz10 = self.slope_z[y0 * self.grid_size + x1];
        let sz01 = self.slope_z[y1 * self.grid_size + x0];
        let sz11 = self.slope_z[y1 * self.grid_size + x1];
        let sz = (sz00 * (1.0 - fx) + sz10 * fx) * (1.0 - fy) + (sz01 * (1.0 - fx) + sz11 * fx) * fy;

        Vec3::new(-sx, 1.0, -sz).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::{H0Spectrum, OceanSpectrumConfig};

    #[test]
    fn test_fft_solve() {
        let config = OceanSpectrumConfig {
            grid_size: 32,
            patch_size: 4.0,
            ..Default::default()
        };
        let spectrum = H0Spectrum::new(config);
        let frame = spectrum.evaluate(0.0);

        let mut solver = OceanFftSolver::new(32);
        let field = solver.solve(&frame, 4.0);

        assert_eq!(field.height.len(), 32 * 32);
        assert_eq!(field.normals.len(), 32 * 32);
        assert!(field.slope_variance > 0.0);

        // Bilinear interpolation test
        let h_sample = field.sample_height_bilinear(2.0, 2.0);
        assert!(h_sample.is_finite());

        let norm_sample = field.sample_normal(2.0, 2.0);
        assert!((norm_sample.length() - 1.0).abs() < 1e-4);
    }
}
