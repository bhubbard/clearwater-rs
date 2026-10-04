use glam::Vec3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use image::{ImageBuffer, Rgb};

/// Mineral color palette in sRGB [0..1] for realistic riverbed and seabed stones.
pub const PALETTE: [[f32; 3]; 12] = [
    [0.62, 0.62, 0.61],
    [0.50, 0.51, 0.53],
    [0.42, 0.44, 0.47],
    [0.66, 0.63, 0.57],
    [0.58, 0.53, 0.45],
    [0.52, 0.47, 0.40],
    [0.36, 0.37, 0.38],
    [0.72, 0.70, 0.66],
    [0.47, 0.43, 0.39],
    [0.55, 0.57, 0.60],
    [0.60, 0.50, 0.42],
    [0.82, 0.80, 0.76],
];

/// Relative weights for palette color selection.
pub const PALETTE_WEIGHTS: [f32; 12] = [
    12.0, 8.0, 6.0, 10.0, 6.0, 4.0, 3.0, 7.0, 3.0, 5.0, 1.5, 3.0,
];

/// Procedurally packed individual stone descriptor.
#[derive(Debug, Clone)]
pub struct Stone {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub aspect: f32,
    pub angle: f32,
    pub z_base: f32,
    pub color_idx: usize,
    pub color_mult: f32,
    pub has_vein: bool,
    pub vein_angle: f32,
    pub vein_offset: f32,
}

/// Seamless procedural pebble seabed texture generator on a 2-torus.
pub struct PebbleBedGenerator {
    pub size: usize,
    pub seed: u64,
}

impl PebbleBedGenerator {
    pub fn new(size: usize, seed: u64) -> Self {
        Self { size, seed }
    }

    /// Generates a photoreal seamless pebble bed texture matching Aurélien's `make_pebbles.py`.
    pub fn generate(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let s = self.size;
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut height = vec![-1.0f32; s * s];
        let mut stone_id = vec![-1i32; s * s];
        let mut stones: Vec<Stone> = Vec::new();

        // 1. Dart-throwing packing on a 2-torus (big stones to small stones)
        let mut pack_layer = |sizes: &[(f32, usize)], overlap: f32, zbase: f32| {
            let mut placed: Vec<(f32, f32, f32)> = Vec::new();

            for &(rmax, tries) in sizes {
                for _ in 0..tries {
                    let r = rmax * rng.gen_range(0.72..1.0);
                    let x = rng.gen_range(0.0..(s as f32));
                    let y = rng.gen_range(0.0..(s as f32));

                    // Check distance on torus
                    let mut ok = true;
                    for &(sx, sy, sr) in &placed {
                        let mut dx = (x - sx).abs();
                        if dx > (s as f32) - dx {
                            dx = (s as f32) - dx;
                        }
                        let mut dy = (y - sy).abs();
                        if dy > (s as f32) - dy {
                            dy = (s as f32) - dy;
                        }
                        let min_dist = overlap * (r + sr);
                        if dx * dx + dy * dy < min_dist * min_dist {
                            ok = false;
                            break;
                        }
                    }

                    if ok {
                        placed.push((x, y, r));
                        let has_vein = rng.gen_bool(0.06);
                        stones.push(Stone {
                            x,
                            y,
                            r,
                            aspect: rng.gen_range(0.5..0.95),
                            angle: rng.gen_range(0.0..std::f32::consts::PI),
                            z_base: zbase + rng.gen_range(0.0..0.08),
                            color_idx: sample_palette_index(&mut rng),
                            color_mult: rng.gen_range(0.88..1.1),
                            has_vein,
                            vein_angle: rng.gen_range(0.0..std::f32::consts::PI),
                            vein_offset: rng.gen_range(0.0..6.28),
                        });
                    }
                }
            }
        };

        // Layer 0: buried bottom layer
        pack_layer(
            &[(30.0, 1500), (22.0, 3000), (15.0, 5000), (10.0, 5000)],
            0.80,
            -0.35,
        );
        // Layer 1: prominent top layer
        pack_layer(
            &[(46.0, 300), (36.0, 900), (28.0, 2500), (21.0, 4000), (15.0, 5000)],
            0.88,
            0.0,
        );

        // 2. Rasterize stones into heightfield and ID buffer
        for (i, stone) in stones.iter().enumerate() {
            let a = stone.r * 1.12;
            let b = a * stone.aspect;
            let r_bound = (a + 3.0).ceil() as i32;

            let cos_ang = stone.angle.cos();
            let sin_ang = stone.angle.sin();
            let p_exp = 2.0 + rng.gen_range(-0.3..0.9);

            let phase1 = rng.gen_range(0.0..6.3);
            let phase2 = rng.gen_range(0.0..6.3);
            let phase3 = rng.gen_range(0.0..6.3);

            let center_x = stone.x as i32;
            let center_y = stone.y as i32;

            for dy in -r_bound..=r_bound {
                let py = (center_y + dy).rem_euclid(s as i32) as usize;
                let cur_dy = (center_y + dy) as f32 - stone.y;

                for dx in -r_bound..=r_bound {
                    let px = (center_x + dx).rem_euclid(s as i32) as usize;
                    let cur_dx = (center_x + dx) as f32 - stone.x;

                    let u = (cur_dx * cos_ang + cur_dy * sin_ang) / a;
                    let v = (-cur_dx * sin_ang + cur_dy * cos_ang) / b;

                    let th = v.atan2(u);
                    let lob = 1.0
                        + 0.07 * (2.0 * th + phase1).sin()
                        + 0.05 * (3.0 * th + phase2).sin()
                        + 0.03 * (5.0 * th + phase3).sin();

                    let d2 = ((u.abs().powf(p_exp) + v.abs().powf(p_exp)).powf(2.0 / p_exp)) / (lob * lob);
                    if d2 < 1.0 {
                        let h = (1.0 - d2).max(0.0).sqrt().powf(0.65)
                            * (0.55 + 0.45 * stone.aspect)
                            * (stone.r / 40.0)
                            + stone.z_base * 0.05;

                        let idx = py * s + px;
                        if h > height[idx] {
                            height[idx] = h;
                            stone_id[idx] = i as i32;
                        }
                    }
                }
            }
        }

        // 3. Shading from height, directional sunlight, ambient occlusion, and shadow
        let light_dir = Vec3::new(-0.45, -0.55, 0.70).normalize();
        let mut img = ImageBuffer::new(s as u32, s as u32);

        for y in 0..s {
            let ym = (y + s - 1) % s;
            let yp = (y + 1) % s;

            for x in 0..s {
                let xm = (x + s - 1) % s;
                let xp = (x + 1) % s;

                let idx = y * s + x;
                let id = stone_id[idx];

                // Base albedo
                let albedo = if id >= 0 {
                    let st = &stones[id as usize];
                    let pal = PALETTE[st.color_idx];
                    let mut col = Vec3::new(pal[0], pal[1], pal[2]) * st.color_mult;

                    // Vein check
                    if st.has_vein {
                        let proj = (x as f32) * st.vein_angle.cos() + (y as f32) * st.vein_angle.sin();
                        let cx = st.x * st.vein_angle.cos() + st.y * st.vein_angle.sin();
                        if (proj - cx - (st.vein_offset - 3.1) * 3.0).abs() < 2.0 {
                            col = col * 0.55 + Vec3::new(0.85, 0.84, 0.80) * 0.45;
                        }
                    }
                    col
                } else {
                    // Dark sand / grit
                    Vec3::new(0.22 * 1.05, 0.22, 0.22 * 0.9)
                };

                // Surface normal from height gradient
                let h_l = height[y * s + xm];
                let h_r = height[y * s + xp];
                let h_d = height[ym * s + x];
                let h_u = height[yp * s + x];

                let gx = (h_r - h_l) * 0.5;
                let gy = (h_u - h_d) * 0.5;

                let k = 14.0f32;
                let norm = Vec3::new(-gx * k, -gy * k, 1.0).normalize();

                // Diffuse sunlight
                let diff = norm.dot(Vec3::new(-light_dir.x, -light_dir.y, light_dir.z)).max(0.0);

                // Ambient occlusion approximation
                let cur_h = height[idx];
                let ao = (0.55 + cur_h * 2.2).clamp(0.35, 1.15);

                // Simple self-shadowing roll offset
                let sh_x = (x + 5) % s;
                let sh_y = (y + 6) % s;
                let shadow_h = height[sh_y * s + sh_x];
                let shadow = (1.0 - (shadow_h - cur_h).max(0.0) * 3.5).clamp(0.40, 1.0);

                let shade = (0.28 + 1.0 * diff) * ao * shadow;
                let spec = norm.dot(Vec3::new(0.2, 0.3, 0.93)).max(0.0).powf(40.0) * 0.08;

                let final_rgb = (albedo * shade + Vec3::splat(spec)) * 1.32;

                let r_byte = (final_rgb.x.clamp(0.0, 1.0) * 255.0) as u8;
                let g_byte = (final_rgb.y.clamp(0.0, 1.0) * 255.0) as u8;
                let b_byte = (final_rgb.z.clamp(0.0, 1.0) * 255.0) as u8;

                img.put_pixel(x as u32, y as u32, Rgb([r_byte, g_byte, b_byte]));
            }
        }

        img
    }
}

fn sample_palette_index(rng: &mut StdRng) -> usize {
    let total_weight: f32 = PALETTE_WEIGHTS.iter().sum();
    let mut choice = rng.gen_range(0.0..total_weight);
    for (i, &w) in PALETTE_WEIGHTS.iter().enumerate() {
        if choice < w {
            return i;
        }
        choice -= w;
    }
    0
}

/// Evaluates shallow water seabed floor depth $y = -D(x, z)$ with shelving shoreline.
pub fn floor_depth(x: f32, z: f32) -> f32 {
    let shelf = 0.95 + 0.17 * (-z + 1.5).clamp(0.0, 14.0);
    let ripple1 = 0.30 * ((x * 0.22).sin() * (z * 0.22).cos());
    let ripple2 = 0.10 * ((x * 0.9 + 7.0).cos() * (z * 0.9).sin());
    shelf + ripple1 + ripple2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pebble_bed_generation() {
        let generator = PebbleBedGenerator::new(64, 11);
        let img = generator.generate();
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
        let pixel = img.get_pixel(32, 32);
        assert!(pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0);
    }

    #[test]
    fn test_floor_depth() {
        let d0 = floor_depth(0.0, 0.0);
        let d_deep = floor_depth(0.0, -10.0);
        assert!(d0 > 0.0);
        assert!(d_deep > d0); // Shore slopes deeper towards open water
    }
}
