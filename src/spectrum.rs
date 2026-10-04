use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// Configuration parameters for the Tessendorf ocean wave spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OceanSpectrumConfig {
    /// Grid resolution $N \times N$, must be a power of two (typically 256).
    pub grid_size: usize,
    /// Physical dimension of the repeating ocean tile in meters (e.g. 4.6 m).
    pub patch_size: f32,
    /// Mean water depth in meters (e.g. 1.6 m).
    pub depth: f32,
    /// Target root-mean-square (RMS) wave slope for realistic wave steepness (e.g. 0.078).
    pub target_slope: f32,
    /// Wind direction unit vector `(wx, wz)` (e.g. `[0.8, 0.6]`).
    pub wind_direction: [f32; 2],
    /// Peak wave wavenumber $k_p = 2\pi / \lambda_p$ (default ~ 2pi / 0.62).
    pub kp: f32,
    /// High-frequency cutoff wavenumber $k_{cut} = 2\pi / \lambda_{cut}$ (default ~ 2pi / 0.045).
    pub kcut: f32,
    /// Gravitational acceleration in $m/s^2$ (9.81).
    pub gravity: f32,
    /// Water surface tension coefficient in $N/m$ (~ 7.4e-5).
    pub surface_tension: f32,
    /// Loop period in seconds over which the dispersion relation is quantized (default 60.0).
    pub loop_period: f32,
    /// Random seed for initial Gaussian noise.
    pub seed: u64,
}

impl Default for OceanSpectrumConfig {
    fn default() -> Self {
        use std::f32::consts::PI;
        Self {
            grid_size: 256,
            patch_size: 4.6,
            depth: 1.6,
            target_slope: 0.078,
            wind_direction: [0.8, 0.6],
            kp: 2.0 * PI / 0.62,
            kcut: 2.0 * PI / 0.045,
            gravity: 9.81,
            surface_tension: 7.4e-5,
            loop_period: 60.0,
            seed: 7,
        }
    }
}

/// Precomputed initial Fourier amplitudes $\tilde{h}_0(\mathbf{k})$ for Tessendorf wave synthesis.
#[derive(Debug, Clone)]
pub struct H0Spectrum {
    pub config: OceanSpectrumConfig,
    /// Flattened $N \times N$ initial positive-frequency complex amplitudes $\tilde{h}_0(\mathbf{k})$.
    pub h0: Vec<Complex32>,
    /// Flattened $N \times N$ conjugate complex amplitudes $\tilde{h}_0^*(-\mathbf{k})$.
    pub h0_conj: Vec<Complex32>,
    /// Precomputed angular frequencies $\omega(\mathbf{k})$.
    pub omega: Vec<f32>,
    /// Precomputed wavenumbers $k_x$ and $k_z$ for each grid point.
    pub kx: Vec<f32>,
    pub kz: Vec<f32>,
}

impl H0Spectrum {
    /// Generates initial conjugate-symmetric Fourier amplitudes from the directional wave spectrum.
    pub fn new(config: OceanSpectrumConfig) -> Self {
        use std::f32::consts::PI;
        let n = config.grid_size;
        let l = config.patch_size;
        let mut rng = StdRng::seed_from_u64(config.seed);

        let mut re = vec![0.0f32; n * n];
        let mut im = vec![0.0f32; n * n];
        let mut s2 = 0.0f32;

        let wd_norm = (config.wind_direction[0].powi(2) + config.wind_direction[1].powi(2)).sqrt();
        let wd = [
            config.wind_direction[0] / wd_norm.max(1e-6),
            config.wind_direction[1] / wd_norm.max(1e-6),
        ];

        // 1. Generate Gaussian random samples shaped by the Phillips / JONSWAP-style spectrum
        for m in 0..n {
            let nz = if m < n / 2 { m as f32 } else { (m as f32) - (n as f32) };
            let kz_val = 2.0 * PI * nz / l;

            for n_idx in 0..n {
                let nx = if n_idx < n / 2 { n_idx as f32 } else { (n_idx as f32) - (n as f32) };
                let kx_val = 2.0 * PI * nx / l;
                let k_hypot = kx_val.hypot(kz_val);

                let mut p = 0.0f32;
                if k_hypot > 1e-6 {
                    let lk = (k_hypot / config.kp).ln();
                    let bump = (-0.5 * (lk / 0.36).powi(2)).exp();
                    let tail = 0.035 * (-(config.kp / k_hypot).powi(2)).exp()
                        * (-(k_hypot / config.kcut).powi(2)).exp();
                    let swell_k = 2.0 * PI / 1.6;
                    let swell = 0.35 * (-0.5 * ((k_hypot / swell_k).ln() / 0.3).powi(2)).exp();

                    let c = (kx_val * wd[0] + kz_val * wd[1]) / k_hypot;
                    let spread = (0.3 + 0.7 * c * c) * if c < 0.0 { 0.35 } else { 1.0 };
                    p = (bump + tail + swell) * spread / (k_hypot.powi(4));
                }

                let amplitude = (p / 2.0).sqrt();
                let g_re: f32 = StandardNormal.sample(&mut rng);
                let g_im: f32 = StandardNormal.sample(&mut rng);

                let idx = m * n + n_idx;
                re[idx] = g_re * amplitude;
                im[idx] = g_im * amplitude;

                s2 += 2.0 * k_hypot * k_hypot * (re[idx] * re[idx] + im[idx] * im[idx]);
            }
        }

        // 2. Scale amplitudes to precisely achieve the target wave RMS slope
        let slope_scale = if s2 > 1e-12 {
            config.target_slope / s2.sqrt()
        } else {
            1.0
        };

        let mut h0 = Vec::with_capacity(n * n);
        let mut h0_conj = Vec::with_capacity(n * n);
        let mut omega = Vec::with_capacity(n * n);
        let mut kx = Vec::with_capacity(n * n);
        let mut kz = Vec::with_capacity(n * n);

        let w0 = if config.loop_period > 0.0 {
            2.0 * PI / config.loop_period
        } else {
            0.0
        };

        for m in 0..n {
            let nz = if m < n / 2 { m as f32 } else { (m as f32) - (n as f32) };
            let kz_val = 2.0 * PI * nz / l;

            for n_idx in 0..n {
                let nx = if n_idx < n / 2 { n_idx as f32 } else { (n_idx as f32) - (n as f32) };
                let kx_val = 2.0 * PI * nx / l;
                let k_hypot = kx_val.hypot(kz_val);

                let i = m * n + n_idx;
                let j = ((n - m) % n) * n + ((n - n_idx) % n);

                h0.push(Complex32::new(re[i] * slope_scale, im[i] * slope_scale));
                h0_conj.push(Complex32::new(re[j] * slope_scale, -im[j] * slope_scale));

                // Shallow water dispersion with surface tension:
                // omega^2 = (g * k + sigma * k^3) * tanh(k * depth)
                let tanh_factor = if config.depth > 0.0 {
                    (k_hypot * config.depth).tanh()
                } else {
                    1.0
                };
                let w_sq = (config.gravity * k_hypot + config.surface_tension * k_hypot.powi(3)) * tanh_factor;
                let mut w = w_sq.max(0.0).sqrt();

                // Quantize frequency so the simulation loops cleanly
                if w0 > 0.0 {
                    w = (w / w0).floor() * w0;
                }

                omega.push(w);
                kx.push(kx_val);
                kz.push(kz_val);
            }
        }

        Self {
            config,
            h0,
            h0_conj,
            omega,
            kx,
            kz,
        }
    }

    /// Evaluates the wave spectrum Fourier components at time `t`.
    pub fn evaluate(&self, t: f32) -> SpectrumFrame {
        let n = self.config.grid_size;
        let mut height_k = Vec::with_capacity(n * n);
        let mut slope_x_k = Vec::with_capacity(n * n);
        let mut slope_z_k = Vec::with_capacity(n * n);
        let mut disp_x_k = Vec::with_capacity(n * n);
        let mut disp_z_k = Vec::with_capacity(n * n);

        for i in 0..(n * n) {
            let w = self.omega[i];
            let wt = w * t;
            let (sin_wt, cos_wt) = wt.sin_cos();

            // e^{i w t} and e^{-i w t}
            let exp_pos = Complex32::new(cos_wt, sin_wt);
            let exp_neg = Complex32::new(cos_wt, -sin_wt);

            // h(k, t) = h0(k) * e^{i w t} + h0*(-k) * e^{-i w t}
            let h = self.h0[i] * exp_pos + self.h0_conj[i] * exp_neg;
            height_k.push(h);

            let kx = self.kx[i];
            let kz = self.kz[i];
            let k_len = kx.hypot(kz);

            // Slopes in Fourier domain: i * k_x * h and i * k_z * h
            // (Note: in spatial domain, d/dx h <-> i * k_x * H)
            let i_unit = Complex32::new(0.0, 1.0);
            slope_x_k.push(i_unit * kx * h);
            slope_z_k.push(i_unit * kz * h);

            // Choppy displacements: -i * (k / |k|) * h
            if k_len > 1e-6 {
                let factor_x = -i_unit * (kx / k_len);
                let factor_z = -i_unit * (kz / k_len);
                disp_x_k.push(factor_x * h);
                disp_z_k.push(factor_z * h);
            } else {
                disp_x_k.push(Complex32::default());
                disp_z_k.push(Complex32::default());
            }
        }

        SpectrumFrame {
            grid_size: n,
            time: t,
            height_k,
            slope_x_k,
            slope_z_k,
            disp_x_k,
            disp_z_k,
        }
    }
}

/// Instantaneous wave Fourier coefficients ready for inverse 2D FFT.
#[derive(Debug, Clone)]
pub struct SpectrumFrame {
    pub grid_size: usize,
    pub time: f32,
    pub height_k: Vec<Complex32>,
    pub slope_x_k: Vec<Complex32>,
    pub slope_z_k: Vec<Complex32>,
    pub disp_x_k: Vec<Complex32>,
    pub disp_z_k: Vec<Complex32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_creation() {
        let config = OceanSpectrumConfig {
            grid_size: 64,
            patch_size: 4.0,
            ..Default::default()
        };
        let spectrum = H0Spectrum::new(config);
        assert_eq!(spectrum.h0.len(), 64 * 64);
        assert_eq!(spectrum.omega.len(), 64 * 64);

        // Check origin k=0 has zero or finite amplitude
        assert!(spectrum.omega[0] >= 0.0);

        let frame = spectrum.evaluate(0.0);
        assert_eq!(frame.height_k.len(), 64 * 64);
        assert_eq!(frame.slope_x_k.len(), 64 * 64);
    }

    #[test]
    fn test_spectrum_temporal_evolution() {
        let config = OceanSpectrumConfig {
            grid_size: 32,
            ..Default::default()
        };
        let spectrum = H0Spectrum::new(config);
        let frame0 = spectrum.evaluate(0.0);
        let frame1 = spectrum.evaluate(1.0);

        // Frames at different times should differ
        let mut diff_sum = 0.0;
        for i in 0..frame0.height_k.len() {
            diff_sum += (frame0.height_k[i] - frame1.height_k[i]).norm();
        }
        assert!(diff_sum > 0.0);
    }
}
