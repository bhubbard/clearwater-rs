use axum::{
    extract::Query,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::info;

use crate::export::HtmlExporter;
use crate::seabed::PebbleBedGenerator;
use crate::spectrum::{H0Spectrum, OceanSpectrumConfig};
use crate::fft::OceanFftSolver;
use crate::caustics::{CausticConfig, CausticEngine};
use glam::Vec3;

#[derive(Debug, Deserialize)]
pub struct QueryParams {
    pub debug: Option<bool>,
    pub t: Option<f32>,
    pub noglare: Option<bool>,
}

pub struct ServerState {
    pub html_content: String,
    pub spectrum: H0Spectrum,
}

/// Runs the local interactive Axum preview server.
pub async fn run_server(port: u16, open_browser: bool) -> Result<(), Box<dyn std::error::Error>> {
    let html_content = HtmlExporter::generate_html();
    let spectrum_config = OceanSpectrumConfig::default();
    let spectrum = H0Spectrum::new(spectrum_config);

    let state = Arc::new(ServerState {
        html_content,
        spectrum,
    });

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/api/status", get(serve_status))
        .route("/api/pebbles.png", get(serve_pebbles_png))
        .route("/api/caustics.png", get(serve_caustics_png))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    info!("Starting Clearwater server on http://{}", addr);

    if open_browser {
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open")
            .arg(format!("http://{}", addr))
            .spawn();

        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open")
            .arg(format!("http://{}", addr))
            .spawn();
    }

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn serve_index(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
    Query(_params): Query<QueryParams>,
) -> Html<String> {
    Html(state.html_content.clone())
}

async fn serve_status() -> Response {
    let payload = serde_json::json!({
        "engine": "clearwater-rs",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "active",
        "features": [
            "Tessendorf 2D FFT Ocean Wave Spectrum",
            "Evan Wallace Refracted-Grid Caustics with Chromatic Dispersion",
            "2D Wave Equation Interactive Ripples",
            "Marc Olano & Dan Baker LEAN Antialiased Glints",
            "Procedural Pebble Seabed on Torus",
            "Beer-Lambert Spectral Absorption"
        ]
    });

    (
        [(header::CONTENT_TYPE, "application/json")],
        payload.to_string(),
    )
        .into_response()
}

async fn serve_pebbles_png() -> Result<Response, StatusCode> {
    let generator = PebbleBedGenerator::new(512, 11);
    let img = generator.generate();

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    img.write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response())
}

async fn serve_caustics_png(
    axum::extract::State(state): axum::extract::State<Arc<ServerState>>,
) -> Result<Response, StatusCode> {
    let frame = state.spectrum.evaluate(0.0);
    let mut solver = OceanFftSolver::new(state.spectrum.config.grid_size);
    let wave = solver.solve(&frame, state.spectrum.config.patch_size);

    let caustic_cfg = CausticConfig {
        ray_grid_size: 128,
        caustic_res: 256,
        patch_size: state.spectrum.config.patch_size,
        depth: state.spectrum.config.depth,
    };
    let engine = CausticEngine::new(caustic_cfg);
    let sun = Vec3::new(0.5, 0.8, -0.3).normalize();
    let map = engine.render(&wave, sun);

    let mut img = image::ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let col = map.rgb[y * 256 + x];
            let r = (col.x * 25.0).clamp(0.0, 255.0) as u8;
            let g = (col.y * 25.0).clamp(0.0, 255.0) as u8;
            let b = (col.z * 25.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    img.write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(([(header::CONTENT_TYPE, "image/png")], bytes).into_response())
}
