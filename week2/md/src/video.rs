//! MP4 video rendering: g(r) computation, PPM frame drawing, and ffmpeg.

use crate::io::{read_output, TrajFrame};
use std::path::Path;
use std::process::Command as ProcessCommand;

pub struct VideoReport {
    pub frames: usize,
    pub out_path: String,
    pub size_bytes: u64,
}

fn min_image(d: f64, l: f64) -> f64 {
    d - l * (d / l).round()
}

// Running (cumulative) radial distribution function: one g(r) curve per
// frame, accumulating pair counts over all frames up to and including the
// current one. Normalized so g(r) -> 1 for an ideal gas.
pub fn cumulative_g(frames: &[TrajFrame], box_len: [f64; 2], dr: f64) -> Vec<Vec<(f64, f64)>> {
    let rmax = box_len[0].min(box_len[1]) / 2.0;
    let n_bins = (rmax / dr).ceil() as usize;
    let area = box_len[0] * box_len[1];

    let mut counts = vec![0usize; n_bins];
    let mut pair_total = 0usize;
    let mut out = Vec::with_capacity(frames.len());

    for frame in frames {
        let n = frame.pos.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = min_image(frame.pos[j][0] - frame.pos[i][0], box_len[0]);
                let dy = min_image(frame.pos[j][1] - frame.pos[i][1], box_len[1]);
                let r = (dx * dx + dy * dy).sqrt();
                if r < rmax {
                    let bin = (r / dr) as usize;
                    if bin < n_bins {
                        counts[bin] += 1;
                        pair_total += 1;
                    }
                }
            }
        }

        let mut g = Vec::with_capacity(n_bins);
        for bin in 0..n_bins {
            let r_lo = bin as f64 * dr;
            let r_hi = r_lo + dr;
            let annulus = std::f64::consts::PI * (r_hi * r_hi - r_lo * r_lo);
            let expected = pair_total as f64 * annulus / area;
            let value = if expected > 0.0 {
                counts[bin] as f64 / expected
            } else {
                0.0
            };
            g.push((r_lo + dr / 2.0, value));
        }
        out.push(g);
    }
    out
}

fn set_px(px: &mut [u8], w: usize, x: usize, y: usize, rgb: [u8; 3]) {
    let h = px.len() / (3 * w);
    if x < w && y < h {
        let idx = (y * w + x) * 3;
        px[idx] = rgb[0];
        px[idx + 1] = rgb[1];
        px[idx + 2] = rgb[2];
    }
}

fn draw_dot(px: &mut [u8], w: usize, x: usize, y: usize, rgb: [u8; 3]) {
    let r = 3isize;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let xx = x as isize + dx;
                let yy = y as isize + dy;
                if xx >= 0 && yy >= 0 {
                    set_px(px, w, xx as usize, yy as usize, rgb);
                }
            }
        }
    }
}

fn draw_line(px: &mut [u8], w: usize, x0: usize, y0: usize, x1: usize, y1: usize, rgb: [u8; 3]) {
    let (mut x, mut y) = (x0 as isize, y0 as isize);
    let (x1, y1) = (x1 as isize, y1 as isize);
    let dx = (x1 - x).abs();
    let sx = if x < x1 { 1 } else { -1 };
    let dy = -(y1 - y).abs();
    let sy = if y < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x >= 0 && y >= 0 {
            set_px(px, w, x as usize, y as usize, rgb);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn render_frame(
    frame: &TrajFrame,
    box_len: [f64; 2],
    g: &[(f64, f64)],
    w: usize,
    h: usize,
    rmax: f64,
) -> Vec<u8> {
    let mut px = vec![255u8; w * h * 3]; // white background
    let margin = 20usize;
    let half = w / 2;

    // Left panel: particle positions in the box.
    let left_w = half - 2 * margin;
    let left_h = h - 2 * margin;
    for p in &frame.pos {
        let sx = margin + (p[0] / box_len[0] * left_w as f64) as usize;
        let sy = margin + (p[1] / box_len[1] * left_h as f64) as usize;
        draw_dot(&mut px, w, sx, sy, [0, 0, 0]);
    }

    // Right panel: g(r) plot.
    let right_x0 = half + margin;
    let right_w = half - 2 * margin;
    let right_h = h - 2 * margin;
    let gmax = 3.0;
    let x_axis_y = h - margin;
    let y_axis_x = right_x0;
    // Axes.
    draw_line(&mut px, w, y_axis_x, margin, y_axis_x, x_axis_y, [0, 0, 0]);
    draw_line(&mut px, w, y_axis_x, x_axis_y, right_x0 + right_w, x_axis_y, [0, 0, 0]);
    // g = 1 reference line.
    let y1 = x_axis_y - (1.0 / gmax * right_h as f64) as usize;
    draw_line(&mut px, w, y_axis_x, y1, right_x0 + right_w, y1, [180, 180, 180]);
    // Curve.
    for wnd in g.windows(2) {
        let x0 = y_axis_x + (wnd[0].0 / rmax * right_w as f64) as usize;
        let y0 = x_axis_y - (wnd[0].1.clamp(0.0, gmax) / gmax * right_h as f64) as usize;
        let x1 = y_axis_x + (wnd[1].0 / rmax * right_w as f64) as usize;
        let y1 = x_axis_y - (wnd[1].1.clamp(0.0, gmax) / gmax * right_h as f64) as usize;
        draw_line(&mut px, w, x0, y0, x1, y1, [200, 0, 0]);
    }
    px
}

fn write_ppm(path: &Path, width: usize, height: usize, pixels: &[u8]) -> std::io::Result<()> {
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    out.extend_from_slice(pixels);
    std::fs::write(path, out)
}

pub fn render_video(dir: &Path, out: &Path) -> Result<VideoReport, String> {
    let output = read_output(dir)?;
    let box_len = output.meta.box_len;
    let rmax = box_len[0].min(box_len[1]) / 2.0;
    let gs = cumulative_g(&output.frames, box_len, 0.05);

    let tmp = std::env::temp_dir().join(format!("md_video_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;

    let w = 960usize;
    let h = 480usize;
    for (idx, (frame, g)) in output.frames.iter().zip(gs.iter()).enumerate() {
        let px = render_frame(frame, box_len, g, w, h, rmax);
        let path = tmp.join(format!("frame_{:05}.ppm", idx + 1));
        write_ppm(&path, w, h, &px).map_err(|e| e.to_string())?;
    }

    let status = ProcessCommand::new("ffmpeg")
        .args(["-y", "-framerate", "10", "-i"])
        .arg(tmp.join("frame_%05d.ppm"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-movflags", "+faststart"])
        .arg(out)
        .status()
        .map_err(|e| format!("failed to run ffmpeg (is it installed?): {e}"))?;
    if !status.success() {
        return Err("ffmpeg exited with an error".to_string());
    }

    let size = std::fs::metadata(out).map_err(|e| e.to_string())?.len();
    let _ = std::fs::remove_dir_all(&tmp);

    if size >= 2_000_000 {
        return Err(format!("video is {size} bytes (>= 2 MB limit)"));
    }

    Ok(VideoReport {
        frames: output.frames.len(),
        out_path: out.display().to_string(),
        size_bytes: size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_peaks_at_known_separation() {
        let frame = TrajFrame {
            step: 50,
            t: 0.5,
            pos: vec![[5.0, 5.0], [6.0, 5.0]],
            vel: vec![[0.0, 0.0], [0.0, 0.0]],
            e_pot: 0.0,
            e_kin: 0.0,
        };
        let box_len = [100.0, 100.0];
        let gs = cumulative_g(&[frame], box_len, 0.05);
        assert_eq!(gs.len(), 1);
        let g = &gs[0];

        let mut peak_idx = 0;
        for (i, (r, _)) in g.iter().enumerate() {
            // `<=` breaks the exact tie between bins 19 and 20 (centers 0.975
            // and 1.025 are equidistant from 1.0) in favor of the higher bin,
            // which is the half-open interval [1.0, 1.05) that contains r=1.0.
            if (r - 1.0).abs() <= (g[peak_idx].0 - 1.0).abs() {
                peak_idx = i;
            }
        }
        for (i, (_, val)) in g.iter().enumerate() {
            if i == peak_idx {
                assert!(*val > 0.0);
            } else {
                assert_eq!(*val, 0.0);
            }
        }
    }

    #[test]
    fn ppm_bytes_have_correct_header() {
        let w = 4usize;
        let h = 2usize;
        let px = vec![0u8; w * h * 3];
        let path = std::env::temp_dir().join("md_test_frame.ppm");
        write_ppm(&path, w, h, &px).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let header = format!("P6\n{w} {h}\n255\n");
        assert!(bytes.starts_with(header.as_bytes()));
        assert_eq!(bytes.len(), header.len() + w * h * 3);
        let _ = std::fs::remove_file(&path);
    }
}
