# Part 4: reproduce the week2 Lennard-Jones fluid artifacts.
# Run from inside week2/:
#   make reproduce
# Builds and runs the `md` crate in release mode with its default run
# parameters (n=100, rho=0.8, temperature=0.5, dt=0.01, eq-steps=2000,
# steps=10000, sample-every=50, seed=2026) and writes run.json + traj.jsonl
# into week2/artifacts/.
.PHONY: reproduce

reproduce:
	cargo run --release --manifest-path md/Cargo.toml --bin md -- run --out artifacts
