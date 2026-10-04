use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;
use tracing_subscriber::EnvFilter;

use clearwater::{
    caustics::{CausticConfig, CausticEngine},
    export::HtmlExporter,
    fft::OceanFftSolver,
    ripples::{RippleDrop, RippleSimulation},
    server::run_server,
    spectrum::{H0Spectrum, OceanSpectrumConfig},
};
use glam::Vec3;

#[derive(Parser)]
#[command(
    name = "clearwater",
    about = "Photoreal shallow water physics and rendering engine in Rust",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the interactive local preview web server
    Serve {
        /// HTTP server port
        #[arg(short, long, default_value_t = 8080)]
        port: u16,

        /// Automatically open browser on start
        #[arg(long, default_value_t = false)]
        open: bool,
    },

    /// Export the self-contained standalone HTML/WebGL2 application
    Export {
        /// Target HTML output path
        #[arg(short, long, default_value = "clearwater.html")]
        output: PathBuf,
    },

    /// Procedurally synthesize a seamless pebble bed texture
    Pebbles {
        /// Target image output path (PNG)
        #[arg(short, long, default_value = "pebbles.png")]
        output: PathBuf,

        /// Texture resolution (must be power of two, e.g. 512, 1024)
        #[arg(short, long, default_value_t = 1024)]
        size: usize,

        /// Random seed
        #[arg(long, default_value_t = 11)]
        seed: u64,
    },

    /// Run the physical water simulation and export heightfield/caustics snapshots
    Sim {
        /// Simulation time in seconds
        #[arg(short, long, default_value_t = 5.0)]
        time: f32,

        /// Output directory for exported frames
        #[arg(short, long, default_value = "sim_output")]
        out_dir: PathBuf,

        /// Grid size resolution
        #[arg(short, long, default_value_t = 256)]
        grid_size: usize,
    },

    /// Benchmark the core physics algorithms (FFT, caustics, ripples)
    Bench {
        /// Number of iterations
        #[arg(short, long, default_value_t = 100)]
        iterations: usize,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, open } => {
            info!("Launching Clearwater interactive preview on port {}", port);
            run_server(port, open).await?;
        }

        Commands::Export { output } => {
            info!("Exporting standalone Clearwater HTML to {:?}", output);
            HtmlExporter::export_to_file(&output)?;
            info!("Export completed successfully: {:?}", output);
        }

        Commands::Pebbles { output, size, seed } => {
            info!("Generating seamless procedural pebble bed ({}x{}, seed: {})...", size, size, seed);
            let start = Instant::now();
            HtmlExporter::export_pebbles_png(&output, size, seed)?;
            info!("Generated {:?} in {:.2?}", output, start.elapsed());
        }

        Commands::Sim { time, out_dir, grid_size } => {
            info!("Running ocean wave simulation at t={}s (grid: {}x{})...", time, grid_size, grid_size);
            std::fs::create_dir_all(&out_dir)?;

            let config = OceanSpectrumConfig {
                grid_size,
                ..Default::default()
            };
            let spectrum = H0Spectrum::new(config);
            let frame = spectrum.evaluate(time);

            let mut solver = OceanFftSolver::new(grid_size);
            let wave = solver.solve(&frame, spectrum.config.patch_size);

            let h_path = out_dir.join(format!("height_t{:.1}.png", time));
            HtmlExporter::export_buffer_png(&h_path, grid_size, grid_size, &wave.height)?;
            info!("Exported wave heightfield to {:?}", h_path);

            let caustic_cfg = CausticConfig {
                ray_grid_size: grid_size,
                caustic_res: grid_size * 2,
                patch_size: spectrum.config.patch_size,
                depth: spectrum.config.depth,
            };
            let engine = CausticEngine::new(caustic_cfg);
            let sun = Vec3::new(0.5, 0.8, -0.3).normalize();
            let caustic_map = engine.render(&wave, sun);

            let mut caustic_gray: Vec<f32> = Vec::with_capacity(caustic_map.resolution * caustic_map.resolution);
            for c in &caustic_map.rgb {
                caustic_gray.push((c.x + c.y + c.z) / 3.0);
            }

            let c_path = out_dir.join(format!("caustics_t{:.1}.png", time));
            HtmlExporter::export_buffer_png(&c_path, caustic_map.resolution, caustic_map.resolution, &caustic_gray)?;
            info!("Exported caustics map to {:?}", c_path);
        }

        Commands::Bench { iterations } => {
            println!("=== Clearwater Engine Performance Benchmark ===");
            let grid_size = 256;
            let config = OceanSpectrumConfig {
                grid_size,
                ..Default::default()
            };

            // 1. Benchmark Spectrum Evolution
            let spectrum = H0Spectrum::new(config);
            let start = Instant::now();
            for i in 0..iterations {
                let _ = spectrum.evaluate(i as f32 * 0.016);
            }
            let dur_spec = start.elapsed();
            println!(
                "Tessendorf Spectrum: {:.2?} total, {:.3?} per frame",
                dur_spec,
                dur_spec / (iterations as u32)
            );

            // 2. Benchmark 2D Inverse FFT
            let frame = spectrum.evaluate(1.0);
            let mut solver = OceanFftSolver::new(grid_size);
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = solver.solve(&frame, 4.6);
            }
            let dur_fft = start.elapsed();
            println!(
                "2D IFFT (5 fields, 256x256): {:.2?} total, {:.3?} per frame ({:.1} FPS)",
                dur_fft,
                dur_fft / (iterations as u32),
                (iterations as f64) / dur_fft.as_secs_f64()
            );

            // 3. Benchmark 2D Ripples
            let mut ripple_sim = RippleSimulation::new(256, 7.0);
            let drop = RippleDrop {
                u: 0.5,
                v: 0.5,
                radius: 0.03,
                strength: 0.1,
            };
            let start = Instant::now();
            for _ in 0..iterations {
                ripple_sim.step(&[drop]);
            }
            let dur_rip = start.elapsed();
            println!(
                "2D Wave Equation Ripples (256x256): {:.2?} total, {:.3?} per step ({:.1} FPS)",
                dur_rip,
                dur_rip / (iterations as u32),
                (iterations as f64) / dur_rip.as_secs_f64()
            );
        }
    }

    Ok(())
}
