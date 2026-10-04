use glam::Vec3;
use std::f32::consts::PI;
use crate::optics::{fresnel, IOR_DEFAULT, SUN_COLOR};

/// LEAN mapping slope covariance representation for anti-aliased ocean highlights.
#[derive(Debug, Clone, Copy)]
pub struct LeanCovariance {
    pub sigma_x2: f32,
    pub sigma_z2: f32,
    pub sigma_xz: f32,
}

impl LeanCovariance {
    /// Computes the effective roughness parameter $a^2$ with slope-variance widening.
    pub fn roughness_sq(&self) -> f32 {
        let trace = 0.5 * (self.sigma_x2 + self.sigma_z2);
        0.00012 + 1.2 * trace
    }

    /// Evaluates Beckmann specular distribution and Smith shadowing/masking for sun glints.
    pub fn evaluate_specular(
        &self,
        normal: Vec3,
        view: Vec3,
        sun_dir: Vec3,
    ) -> Vec3 {
        let h = (view + sun_dir).normalize();
        let nh = normal.dot(h).max(0.0);
        let nl = normal.dot(sun_dir).max(0.0);
        let nv = normal.dot(view).max(0.02);

        let a2 = self.roughness_sq();

        let c2 = (nh * nh).max(1e-4);
        let tan2 = (1.0 - c2) / c2;

        // Beckmann distribution: no long GGX tail, crisp sparkling glints
        let d = (-tan2 / a2).exp() / (PI * a2 * c2 * c2);

        // Kelemen / Smith visibility shadowing-masking term
        let term_l = nl * (nv * nv * (1.0 - a2) + a2).sqrt();
        let term_v = nv * (nl * nl * (1.0 - a2) + a2).sqrt();
        let vis = 0.5 / (term_l + term_v + 1e-5);

        let fh = fresnel(h.dot(view).max(0.0), IOR_DEFAULT);

        let intensity = (d * vis * fh * nl).min(12000.0);
        SUN_COLOR * intensity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lean_glints() {
        let lean = LeanCovariance {
            sigma_x2: 0.05,
            sigma_z2: 0.05,
            sigma_xz: 0.0,
        };
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let view = Vec3::new(0.0, 0.7071, 0.7071).normalize();
        let sun = Vec3::new(0.0, 0.7071, -0.7071).normalize();

        let spec = lean.evaluate_specular(normal, view, sun);
        assert!(spec.x >= 0.0 && spec.x.is_finite());
    }
}
