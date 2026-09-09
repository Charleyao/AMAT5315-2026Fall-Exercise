//! Serde types and file I/O for run.json and traj.jsonl.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RunMeta {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_len: [f64; 2],
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrajFrame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SimulationOutput {
    pub meta: RunMeta,
    pub frames: Vec<TrajFrame>,
}

pub fn write_output(dir: &Path, output: &SimulationOutput) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let run_json = serde_json::to_string_pretty(&output.meta)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(dir.join("run.json"), run_json + "\n")?;

    let mut traj = String::new();
    for frame in &output.frames {
        let line = serde_json::to_string(frame)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        traj.push_str(&line);
        traj.push('\n');
    }
    std::fs::write(dir.join("traj.jsonl"), traj)?;
    Ok(())
}

pub fn read_output(dir: &Path) -> Result<SimulationOutput, String> {
    let run_json = std::fs::read_to_string(dir.join("run.json"))
        .map_err(|e| format!("read run.json: {e}"))?;
    let meta: RunMeta =
        serde_json::from_str(&run_json).map_err(|e| format!("parse run.json: {e}"))?;

    let traj_json = std::fs::read_to_string(dir.join("traj.jsonl"))
        .map_err(|e| format!("read traj.jsonl: {e}"))?;
    let mut frames = Vec::new();
    for (i, line) in traj_json.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let frame: TrajFrame = serde_json::from_str(line)
            .map_err(|e| format!("parse traj.jsonl line {}: {e}", i + 1))?;
        frames.push(frame);
    }
    Ok(SimulationOutput { meta, frames })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_round_trips() {
        let dir = std::env::temp_dir().join("md_io_roundtrip_test");
        let _ = std::fs::remove_dir_all(&dir);
        let output = SimulationOutput {
            meta: RunMeta {
                n: 4,
                rho: 0.8,
                box_len: [1.0, 1.0],
                dt: 0.01,
                temperature: 0.5,
                eq_steps: 0,
                steps: 50,
                sample_every: 50,
                seed: 2026,
                integrator: "velocity-verlet".to_string(),
            },
            frames: vec![TrajFrame {
                step: 50,
                t: 0.5,
                pos: vec![[0.0, 0.0]],
                vel: vec![[1.0, 0.0]],
                e_pot: -1.0,
                e_kin: 0.5,
            }],
        };
        write_output(&dir, &output).unwrap();
        let read = read_output(&dir).unwrap();
        assert_eq!(read.meta.n, 4);
        assert_eq!(read.meta.integrator.as_str(), "velocity-verlet");
        assert_eq!(read.meta.box_len, [1.0, 1.0]);
        assert_eq!(read.frames.len(), 1);
        assert_eq!(read.frames[0].step, 50);
        assert_eq!(read.frames[0].pos.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
