use clearwater::{
    caustics::{CausticConfig, CausticEngine},
    fft::OceanFftSolver,
    lean::LeanCovariance,
    optics::{beer_lambert, fresnel, refract, sky_radiance, IOR_DEFAULT},
    ripples::{RippleDrop, RippleSimulation},
    seabed::{floor_depth, PebbleBedGenerator},
    spectrum::{H0Spectrum, OceanSpectrumConfig},
    HtmlExporter,
};
use glam::Vec3;

#[test]
fn test_end_to_end_water_pipeline() {
    // 1. Spectrum generation
    let config = OceanSpectrumConfig {
        grid_size: 64,
        patch_size: 4.6,
        depth: 1.6,
        target_slope: 0.078,
        ..Default::default()
    };
    let spectrum = H0Spectrum::new(config);
    let frame = spectrum.evaluate(2.5);

    // 2. 2D FFT surface evaluation
    let mut solver = OceanFftSolver::new(64);
    let wave = solver.solve(&frame, 4.6);
    assert_eq!(wave.height.len(), 64 * 64);
    assert_eq!(wave.normals.len(), 64 * 64);

    // 3. Physical optics & caustics
    let sun = Vec3::new(0.4, 0.8, -0.4).normalize();
    let caustic_cfg = CausticConfig {
        ray_grid_size: 64,
        caustic_res: 128,
        patch_size: 4.6,
        depth: 1.6,
    };
    let caustic_engine = CausticEngine::new(caustic_cfg);
    let caustic_map = caustic_engine.render(&wave, sun);
    assert_eq!(caustic_map.rgb.len(), 128 * 128);

    // 4. Interactive ripple simulation
    let mut ripples = RippleSimulation::new(64, 7.0);
    ripples.step(&[RippleDrop {
        u: 0.5,
        v: 0.5,
        radius: 0.04,
        strength: 0.08,
    }]);
    let (h_sample, _, _, _) = ripples.sample_at_world_pos(0.0, 0.0);
    assert!(h_sample.is_finite());

    // 5. LEAN highlight
    let lean = LeanCovariance {
        sigma_x2: wave.slope_variance,
        sigma_z2: wave.slope_variance,
        sigma_xz: 0.0,
    };
    let spec = lean.evaluate_specular(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.7, 0.7).normalize(),
        sun,
    );
    assert!(spec.x >= 0.0);

    // 6. Seabed depth and pebble synthesis
    let seabed_depth = floor_depth(1.0, 2.0);
    assert!(seabed_depth > 0.0);

    let pebble_gen = PebbleBedGenerator::new(64, 11);
    let pebble_img = pebble_gen.generate();
    assert_eq!(pebble_img.width(), 64);

    // 7. Standalone HTML exporter test
    let html = HtmlExporter::generate_html();
    assert!(html.contains("<!doctype html>"));
    assert!(html.contains("Clearwater"));
}

#[test]
fn test_optics_consistency() {
    // Conservation of energy: Fresnel reflection + transmission = 1.0
    let cos_theta = 0.6;
    let r = fresnel(cos_theta, IOR_DEFAULT);
    let t = 1.0 - r;
    assert!(r >= 0.0 && r <= 1.0);
    assert!(t >= 0.0 && t <= 1.0);

    // Refraction follows Snell's Law
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let inc = Vec3::new(0.5, -0.866, 0.0).normalize();
    let refr = refract(inc, normal, 1.0 / IOR_DEFAULT).unwrap();
    let sin_inc = inc.x.abs();
    let sin_refr = refr.x.abs();
    assert!((sin_inc - IOR_DEFAULT * sin_refr).abs() < 1e-4);

    // Beer-Lambert transmittance decreases monotonically with depth
    let t1 = beer_lambert(1.0);
    let t2 = beer_lambert(2.0);
    assert!(t1.x > t2.x);
    assert!(t1.y > t2.y);
    assert!(t1.z > t2.z);

    // Sky radiance is positive
    let sky = sky_radiance(Vec3::new(0.0, 0.5, 0.5).normalize(), Vec3::new(0.2, 0.8, -0.3).normalize());
    assert!(sky.x > 0.0 && sky.y > 0.0 && sky.z > 0.0);
}
