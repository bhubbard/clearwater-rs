use std::fs;
use std::path::Path;
use image::{ImageBuffer, Rgb};
use crate::seabed::PebbleBedGenerator;

/// Exporter for the standalone single-file WebGL2 Clearwater HTML application.
pub struct HtmlExporter;

impl HtmlExporter {
    /// Generates a standalone self-contained HTML file embedding all shaders, simulation logic,
    /// and the procedurally synthesized pebble bed texture.
    pub fn export_to_file<P: AsRef<Path>>(output_path: P) -> std::io::Result<()> {
        let html_content = Self::generate_html();
        fs::write(output_path, html_content)
    }

    /// Generates the full HTML/WebGL2 document string.
    pub fn generate_html() -> String {
        // Upstream HTML document with all shaders, FFT spectrum, ripples, and lens glare
        let upstream_template = include_str!("../templates/index.html");
        upstream_template.to_string()
    }

    /// Exports a procedural pebble bed texture to a PNG file.
    pub fn export_pebbles_png<P: AsRef<Path>>(output_path: P, size: usize, seed: u64) -> std::io::Result<()> {
        let generator = PebbleBedGenerator::new(size, seed);
        let img = generator.generate();
        img.save(output_path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    /// Exports floating-point heightfield or caustic map buffer to an 8-bit PNG image.
    pub fn export_buffer_png<P: AsRef<Path>>(
        output_path: P,
        width: usize,
        height: usize,
        data: &[f32],
    ) -> std::io::Result<()> {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in data {
            if v.is_finite() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }
        let range = if (max_val - min_val).abs() > 1e-6 {
            max_val - min_val
        } else {
            1.0
        };

        let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let val = data[y * width + x];
                let norm = ((val - min_val) / range).clamp(0.0, 1.0);
                let byte = (norm * 255.0) as u8;
                img.put_pixel(x as u32, y as u32, Rgb([byte, byte, byte]));
            }
        }

        img.save(output_path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_buffer() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_buf.png");
        let data = vec![0.5f32; 16 * 16];
        HtmlExporter::export_buffer_png(&path, 16, 16, &data).unwrap();
        assert!(path.exists());
    }
}
