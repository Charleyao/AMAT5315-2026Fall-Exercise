//! Video rendering (implemented in Task 11).

use std::path::Path;

pub struct VideoReport {
    pub frames: usize,
    pub out_path: String,
    pub size_bytes: u64,
}

pub fn render_video(_dir: &Path, _out: &Path) -> Result<VideoReport, String> {
    Err("md video is not implemented yet".to_string())
}
