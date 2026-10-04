use serde::{Deserialize, Serialize};

/// Droplet drop disturbance configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RippleDrop {
    /// Normalized coordinates `[0..1, 0..1]` within the simulation grid.
    pub u: f32,
    pub v: f32,
    /// Radius in normalized UV coordinates (e.g. 0.022).
    pub radius: f32,
    /// Impulse disturbance amplitude (e.g. 0.07).
    pub strength: f32,
}

/// 2D wave equation ripple simulation grid.
#[derive(Debug, Clone)]
pub struct RippleSimulation {
    pub grid_size: usize,
    /// Physical size of simulation window in meters (e.g. 7.0 m).
    pub domain_size: f32,
    /// Current height state grid $H$.
    pub height: Vec<f32>,
    /// Velocity state grid $V$.
    pub velocity: Vec<f32>,
    /// Slopes $\partial h / \partial x$.
    pub slope_x: Vec<f32>,
    /// Slopes $\partial h / \partial z$.
    pub slope_z: Vec<f32>,
    /// Laplacian $\nabla^2 h$ used for caustic light focusing / lensing.
    pub laplacian: Vec<f32>,
    /// Center of the simulation window in world coordinates `(x, z)`.
    pub center: [f32; 2],
    /// Velocity damping factor (default 0.9955).
    pub velocity_damping: f32,
    /// Height damping factor (default 0.9985).
    pub height_damping: f32,
    /// Coupling propagation speed constant (default 0.9).
    pub wave_speed: f32,
}

impl RippleSimulation {
    /// Creates a new ripple simulation grid initialized to equilibrium.
    pub fn new(grid_size: usize, domain_size: f32) -> Self {
        let count = grid_size * grid_size;
        Self {
            grid_size,
            domain_size,
            height: vec![0.0; count],
            velocity: vec![0.0; count],
            slope_x: vec![0.0; count],
            slope_z: vec![0.0; count],
            laplacian: vec![0.0; count],
            center: [0.0, 0.0],
            velocity_damping: 0.9955,
            height_damping: 0.9985,
            wave_speed: 0.9,
        }
    }

    /// Shifts the simulation grid in discrete texel increments when the camera moves.
    pub fn shift_grid(&mut self, dx_texels: i32, dz_texels: i32) {
        if dx_texels == 0 && dz_texels == 0 {
            return;
        }

        let n = self.grid_size as i32;
        let mut new_h = vec![0.0; self.grid_size * self.grid_size];
        let mut new_v = vec![0.0; self.grid_size * self.grid_size];

        for y in 0..n {
            let src_y = y + dz_texels;
            if src_y < 0 || src_y >= n {
                continue;
            }
            for x in 0..n {
                let src_x = x + dx_texels;
                if src_x < 0 || src_x >= n {
                    continue;
                }
                let dst_idx = (y * n + x) as usize;
                let src_idx = (src_y * n + src_x) as usize;
                new_h[dst_idx] = self.height[src_idx];
                new_v[dst_idx] = self.velocity[src_idx];
            }
        }

        self.height = new_h;
        self.velocity = new_v;

        let texel_m = self.domain_size / (self.grid_size as f32);
        self.center[0] += (dx_texels as f32) * texel_m;
        self.center[1] += (dz_texels as f32) * texel_m;
    }

    /// Adds a droplet impulse disturbance at UV coordinates.
    pub fn add_drop(&mut self, drop: RippleDrop) {
        let n = self.grid_size;
        use std::f32::consts::PI;

        for y in 0..n {
            let v = (y as f32) / (n as f32);
            for x in 0..n {
                let u = (x as f32) / (n as f32);
                let du = u - drop.u;
                let dv = v - drop.v;
                let dist = du.hypot(dv);

                if dist < drop.radius && drop.radius > 1e-6 {
                    let factor = 0.5 + 0.5 * (PI * dist / drop.radius).cos();
                    let idx = y * n + x;
                    self.height[idx] -= drop.strength * factor;
                }
            }
        }
    }

    /// Steps the wave equation forward by one discrete time step.
    pub fn step(&mut self, drops: &[RippleDrop]) {
        let n = self.grid_size;
        let mut next_h = self.height.clone();
        let mut next_v = self.velocity.clone();

        for y in 0..n {
            let ym = if y > 0 { y - 1 } else { y };
            let yp = if y + 1 < n { y + 1 } else { y };

            let v_norm = (y as f32) / (n as f32);
            let edge_v = (v_norm.min(1.0 - v_norm) / 0.06).clamp(0.0, 1.0);

            for x in 0..n {
                let xm = if x > 0 { x - 1 } else { x };
                let xp = if x + 1 < n { x + 1 } else { x };

                let u_norm = (x as f32) / (n as f32);
                let edge_u = (u_norm.min(1.0 - u_norm) / 0.06).clamp(0.0, 1.0);
                let edge = edge_u.min(edge_v);

                let idx = y * n + x;
                let cur_h = self.height[idx];
                let cur_v = self.velocity[idx];

                let h_l = self.height[y * n + xm];
                let h_r = self.height[y * n + xp];
                let h_d = self.height[ym * n + x];
                let h_u = self.height[yp * n + x];

                // 2D discrete Laplacian 4-point stencil average
                let avg = 0.25 * (h_l + h_r + h_d + h_u);

                let mut v = cur_v + (avg - cur_h) * self.wave_speed;
                v *= self.velocity_damping;

                let mut h = cur_h + v;
                h *= self.height_damping;

                // Smooth edge fade towards simulation borders
                let edge_blend = 0.9 + 0.1 * edge;
                h *= edge_blend;
                v *= edge_blend;

                next_h[idx] = h;
                next_v[idx] = v;
            }
        }

        self.height = next_h;
        self.velocity = next_v;

        // Apply external drop perturbations
        for drop in drops {
            self.add_drop(*drop);
        }

        // Recompute normals, slopes, and Laplacian
        self.update_derivatives();
    }

    /// Computes spatial derivatives $\partial h / \partial x$, $\partial h / \partial z$ and Laplacian $\nabla^2 h$.
    fn update_derivatives(&mut self) {
        let n = self.grid_size;
        let texel_m = self.domain_size / (n as f32);
        let two_texel = 2.0 * texel_m;
        let texel_sq = texel_m * texel_m;

        for y in 0..n {
            let ym = if y > 0 { y - 1 } else { y };
            let yp = if y + 1 < n { y + 1 } else { y };

            for x in 0..n {
                let xm = if x > 0 { x - 1 } else { x };
                let xp = if x + 1 < n { x + 1 } else { x };

                let idx = y * n + x;
                let h = self.height[idx];
                let h_l = self.height[y * n + xm];
                let h_r = self.height[y * n + xp];
                let h_d = self.height[ym * n + x];
                let h_u = self.height[yp * n + x];

                let hx = (h_r - h_l) / two_texel;
                let hz = (h_u - h_d) / two_texel;
                let lap = (h_l + h_r + h_d + h_u - 4.0 * h) / texel_sq;

                self.slope_x[idx] = hx;
                self.slope_z[idx] = hz;
                self.laplacian[idx] = lap;
            }
        }
    }

    /// Evaluates ripple height and slope at physical coordinates `(wx, wz)` relative to the simulation center.
    pub fn sample_at_world_pos(&self, wx: f32, wz: f32) -> (f32, f32, f32, f32) {
        let u = (wx - self.center[0]) / self.domain_size + 0.5;
        let v = (wz - self.center[1]) / self.domain_size + 0.5;

        if u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let n = self.grid_size as f32;
        let px = (u * n).clamp(0.0, n - 1.0);
        let py = (v * n).clamp(0.0, n - 1.0);

        let x0 = (px.floor() as usize).min(self.grid_size - 1);
        let y0 = (py.floor() as usize).min(self.grid_size - 1);
        let x1 = (x0 + 1).min(self.grid_size - 1);
        let y1 = (y0 + 1).min(self.grid_size - 1);

        let fx = px.fract();
        let fy = py.fract();

        let interp = |grid: &[f32]| {
            let v00 = grid[y0 * self.grid_size + x0];
            let v10 = grid[y0 * self.grid_size + x1];
            let v01 = grid[y1 * self.grid_size + x0];
            let v11 = grid[y1 * self.grid_size + x1];
            let top = v00 * (1.0 - fx) + v10 * fx;
            let bot = v01 * (1.0 - fx) + v11 * fx;
            top * (1.0 - fy) + bot * fy
        };

        (
            interp(&self.height),
            interp(&self.slope_x),
            interp(&self.slope_z),
            interp(&self.laplacian),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ripple_propagation() {
        let mut sim = RippleSimulation::new(64, 7.0);
        let drop = RippleDrop {
            u: 0.5,
            v: 0.5,
            radius: 0.05,
            strength: 0.1,
        };

        sim.add_drop(drop);
        let initial_h = sim.height[32 * 64 + 32];
        assert!(initial_h < 0.0); // Droplet depression

        // Run several simulation steps
        for _ in 0..10 {
            sim.step(&[]);
        }

        // Ripple should spread outwards
        let (h_center, _, _, _) = sim.sample_at_world_pos(0.0, 0.0);
        let (h_neighbor, _, _, _) = sim.sample_at_world_pos(0.5, 0.0);
        assert!(h_center.is_finite());
        assert!(h_neighbor.is_finite());
    }
}
