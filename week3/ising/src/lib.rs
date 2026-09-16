//! Two-dimensional Ising model sampler.
//!
//! Spins are +1/-1 on an L x L periodic square lattice, J = 1, no external
//! field, and temperature is measured in units of J. Two update rules are
//! provided: single-site Metropolis sweeps and Wolff single-cluster flips.

pub mod cli;
pub mod io;
pub mod lattice;
pub mod metropolis;
pub mod simulation;
pub mod wolff;
