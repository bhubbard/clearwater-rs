use glam::Vec3;
use std::f32::consts::PI;

/// Index of refraction for pure liquid water across visible light spectrum.
pub const IOR_DEFAULT: f32 = 1.3335;

/// Chromatic dispersion indices of refraction for RGB primary wavelengths.
pub const IOR_RED: f32 = 1.3315;
pub const IOR_GREEN: f32 = 1.3335;
pub const IOR_BLUE: f32 = 1.3365;

/// Spectral volumetric absorption coefficient $\sigma_a$ in $m^{-1}$ (Red, Green, Blue).
/// Higher red absorption creates the characteristic crystalline turquoise tint.
pub const SIGMA_A: Vec3 = Vec3::new(0.40, 0.074, 0.088);

/// Spectral volumetric scattering coefficient $\sigma_s$ in $m^{-1}$.
pub const SIGMA_S: Vec3 = Vec3::new(0.028, 0.052, 0.068);

/// Extinction coefficient $\sigma_t = \sigma_a + \sigma_s$.
pub const SIGMA_T: Vec3 = Vec3::new(
    SIGMA_A.x + SIGMA_S.x,
    SIGMA_A.y + SIGMA_S.y,
    SIGMA_A.z + SIGMA_S.z,
);

/// Solar illuminance color and intensity multiplier.
pub const SUN_COLOR: Vec3 = Vec3::new(1.0 * 6.0, 0.90 * 6.0, 0.74 * 6.0);

/// Computes physical unpolarized dielectric Fresnel reflection coefficient $F \in [0, 1]$.
pub fn fresnel(cos_theta_i: f32, ior: f32) -> f32 {
    let ci = cos_theta_i.clamp(0.0, 1.0);
    let sin2_t = (1.0 - ci * ci) / (ior * ior);
    if sin2_t >= 1.0 {
        return 1.0; // Total internal reflection
    }
    let ct = (1.0 - sin2_t).sqrt();
    let rs = (ci - ior * ct) / (ci + ior * ct);
    let rp = (ior * ci - ct) / (ior * ci + ct);
    0.5 * (rs * rs + rp * rp)
}

/// Refracts an incident unit direction vector $\mathbf{v}$ through surface normal $\mathbf{n}$ with index of refraction ratio $\eta = \eta_1 / \eta_2$.
pub fn refract(v: Vec3, n: Vec3, eta: f32) -> Option<Vec3> {
    let cos_theta_i = (-v).dot(n);
    let sin2_theta_t = eta * eta * (1.0 - cos_theta_i * cos_theta_i);
    if sin2_theta_t >= 1.0 {
        return None; // Total internal reflection
    }
    let cos_theta_t = (1.0 - sin2_theta_t).sqrt();
    Some(eta * v + (eta * cos_theta_i - cos_theta_t) * n)
}

/// Reflects an incident unit vector $\mathbf{v}$ off a surface with unit normal $\mathbf{n}$.
pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * v.dot(n) * n
}

/// Beer-Lambert transmission attenuation through a column of water of distance $s$ meters.
pub fn beer_lambert(distance: f32) -> Vec3 {
    Vec3::new(
        (-SIGMA_T.x * distance).exp(),
        (-SIGMA_T.y * distance).exp(),
        (-SIGMA_T.z * distance).exp(),
    )
}

/// Henyey-Greenstein single-scattering forward phase function.
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
    if denom > 1e-6 {
        (1.0 - g2) / (4.0 * PI * denom)
    } else {
        0.0
    }
}

/// Computes analytic sky radiance for ray direction `d` and sun direction `sun`.
pub fn sky_radiance(d: Vec3, sun: Vec3) -> Vec3 {
    let elevation = d.y;
    let mu = d.dot(sun);

    let zenith = Vec3::new(0.11, 0.27, 0.62);
    let horizon = Vec3::new(0.66, 0.78, 0.90);

    let blend = elevation.clamp(0.0, 1.0).powf(0.42);
    let mut color = horizon.lerp(zenith, blend);

    // Solar flare and circumsolar aureole
    let mu_pos = mu.max(0.0);
    let sun_glow = Vec3::new(1.0, 0.86, 0.66)
        * (0.22 * mu_pos.powi(6) + 0.30 * mu_pos.powi(64) + 1.6 * mu_pos.powf(2400.0));
    color += sun_glow;

    // Distant headland ridge line
    let azimuth = d.z.atan2(d.x);
    let ridge_elevation = 0.040
        + 0.016 * (azimuth * 2.0 + 0.7).sin()
        + 0.011 * (azimuth * 5.0 + 2.1).sin()
        + 0.006 * (azimuth * 11.0 + 0.3).sin()
        + 0.003 * (azimuth * 23.0 + 1.7).sin();

    if elevation < ridge_elevation && elevation >= -0.3 {
        let pine = Vec3::new(0.045, 0.070, 0.042) * 0.9;
        let rock = Vec3::new(0.30, 0.28, 0.23) * 0.7;
        let land = pine.lerp(rock, 0.5);
        let aerial_perspective = horizon * 0.92;
        let land_color = land.lerp(aerial_perspective, 0.45);

        let t = ((ridge_elevation - elevation) / 0.002).clamp(0.0, 1.0);
        color = color.lerp(land_color, t);
    }

    color
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresnel_properties() {
        // Normal incidence: R = ((1 - n) / (1 + n))^2
        let f0 = fresnel(1.0, IOR_DEFAULT);
        let expected_f0 = ((1.0 - IOR_DEFAULT) / (1.0 + IOR_DEFAULT)).powi(2);
        assert!((f0 - expected_f0).abs() < 1e-4);

        // Grazing incidence: R -> 1.0
        let f_grazing = fresnel(0.0, IOR_DEFAULT);
        assert_eq!(f_grazing, 1.0);
    }

    #[test]
    fn test_snells_law() {
        let incident = Vec3::new(0.0, -1.0, 0.0); // Perpendicular down
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let refracted = refract(incident, normal, 1.0 / IOR_DEFAULT).unwrap();
        assert!((refracted.y - (-1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_beer_lambert_attenuation() {
        let shallow = beer_lambert(0.5);
        let deep = beer_lambert(3.0);
        // Red light should attenuate significantly faster than blue/green
        assert!(shallow.x > deep.x);
        assert!(deep.y > deep.x); // Green transmits better than red in water
    }
}
