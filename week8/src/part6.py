"""AMAT5315 Week 8, Part 6: scaling laws.

Run from the repository root with::

    cd week8 && uv run python -m src.part6
"""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .part6_plotting import (
    plot_figure19,
    plot_figure20,
    plot_figure21,
)
from .scaling_laws import (
    BOOTSTRAP_SEED,
    CHINCHILLA_COMPUTE,
    CHINCHILLA_D,
    CHINCHILLA_N,
    HUBER_DELTA,
    N_BOOTSTRAP,
    PARAMETER_NAMES,
    PUBLISHED_PARAMS,
    REFERENCE_PARAMS,
    bootstrap_fits,
    compute_optimal,
    fit_scaling_law,
    load_chinchilla_data,
    log_residuals,
    ratio_at_compute,
    summarize_bootstrap,
)

WEEK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_DIR / "data"
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
DATA_FILE = DATA_DIR / "chinchilla-runs.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/"
    "92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv"
)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _params_tuple(params: dict) -> tuple:
    return tuple(float(params[name]) for name in PARAMETER_NAMES)


def run() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"missing {DATA_FILE}; run `python -m src.download_part6_data`"
        )

    # ------------------------------------------------------------- data
    data = load_chinchilla_data(DATA_FILE)
    N, D, L = data["N"], data["D"], data["L"]
    print(
        f"[data] {data['n_total']} runs, removed 5 highest-loss rows, "
        f"{data['n_used']} remain"
    )

    # -------------------------------------------------------------- fit
    fit = fit_scaling_law(N, D, L)
    params = fit["params"]
    params_tuple = _params_tuple(params)
    print(
        "[fit ] E={E:.6f}, A={A:.3f}, B={B:.3f}, "
        "alpha={alpha:.6f}, beta={beta:.6f}".format(**params)
    )

    bootstrap = bootstrap_fits(
        N,
        D,
        L,
        fit["vector"],
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
        log=print,
    )
    bootstrap_summary = summarize_bootstrap(bootstrap["draws"])
    bootstrap_ratios = np.array(
        [
            ratio_at_compute(_params_tuple_from_draw(draw))["D_over_N"]
            for draw in bootstrap["draws"]
        ]
    )

    # ------------------------------------------------------- VERIFY (1)
    verify1_checks = {
        "alpha_in_0.32_0.38": 0.32 <= params["alpha"] <= 0.38,
        "beta_in_0.32_0.42": 0.32 <= params["beta"] <= 0.42,
        "E_in_1.75_1.90": 1.75 <= params["E"] <= 1.90,
    }
    verify1_passed = bool(all(verify1_checks.values()))
    figure19_path = plot_figure19(
        N,
        data["C"],
        L,
        D,
        params_tuple,
        EVIDENCE_DIR / "figure19_scaling_fit.png",
    )

    # ------------------------------------------------------- VERIFY (2)
    published_tuple = tuple(
        float(PUBLISHED_PARAMS[name]) for name in PARAMETER_NAMES
    )
    our_residuals = log_residuals(params_tuple, N, D, L)
    published_residuals = log_residuals(published_tuple, N, D, L)
    our_mean_residual = float(our_residuals.mean())
    published_mean_residual = float(published_residuals.mean())
    verify2_checks = {
        "our_abs_mean_residual_le_0.005": abs(our_mean_residual) <= 0.005,
        "published_mean_residual_gt_0.01": published_mean_residual > 0.01,
    }
    verify2_passed = bool(all(verify2_checks.values()))
    figure20_path = plot_figure20(
        N,
        D,
        L,
        params_tuple,
        published_tuple,
        EVIDENCE_DIR / "figure20_residuals.png",
    )

    # ------------------------------------------------------- VERIFY (3)
    our_ratio = ratio_at_compute(params_tuple, CHINCHILLA_COMPUTE)
    published_ratio = ratio_at_compute(
        published_tuple, CHINCHILLA_COMPUTE
    )
    ratio_ci = np.percentile(bootstrap_ratios, [2.5, 97.5])
    ratio_check = 10.0 <= our_ratio["D_over_N"] <= 30.0
    reference_close = (
        abs(our_ratio["D_over_N"] - 17.9) / 17.9 <= 0.05
    )
    chinchilla_difference = (
        our_ratio["D_over_N"] / 20.0 - 1.0
    )
    verify3_checks = {
        "ratio_between_10_and_30": ratio_check,
        "reference_ratio_within_5_percent": reference_close,
    }
    verify3_passed = bool(all(verify3_checks.values()))
    figure21_path = plot_figure21(
        params_tuple,
        bootstrap["draws"],
        EVIDENCE_DIR / "figure21_compute_optimal.png",
    )

    # --------------------------------------------------------- artifacts
    bootstrap_path = ARTIFACT_DIR / "part6_bootstrap.npz"
    np.savez_compressed(
        bootstrap_path,
        draws=bootstrap["draws"],
        objectives=bootstrap["objectives"],
        ratios=bootstrap_ratios,
        parameter_names=np.asarray(PARAMETER_NAMES, dtype=str),
    )

    compute_grid = np.logspace(18, 25, 50)
    n_ours, d_ours = compute_optimal(params_tuple, compute_grid)
    n_pub, d_pub = compute_optimal(published_tuple, compute_grid)
    compute_csv_path = ARTIFACT_DIR / "part6_compute_optimal.csv"
    with compute_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "compute_flop",
                "N_star_ours",
                "D_star_ours",
                "N_star_published",
                "D_star_published",
            ]
        )
        for row in zip(compute_grid, n_ours, d_ours, n_pub, d_pub):
            writer.writerow([f"{value:.12g}" for value in row])

    fitted = params_tuple[0] + params_tuple[1] / N ** params_tuple[3] + (
        params_tuple[2] / D ** params_tuple[4]
    )
    results = {
        "part": "Week 8 Part 6 - scaling laws",
        "dataset": {
            "url": DATA_URL,
            "file": DATA_FILE.name,
            "n_total": data["n_total"],
            "n_used": data["n_used"],
            "removed_highest_loss": data["removed"],
            "N_min": float(N.min()),
            "N_max": float(N.max()),
            "D_min": float(D.min()),
            "D_max": float(D.max()),
            "L_min": float(L.min()),
            "L_max": float(L.max()),
        },
        "fit": {
            "params": {name: float(params[name]) for name in PARAMETER_NAMES},
            "objective": fit["objective"],
            "success": fit["success"],
            "iterations": fit["iterations"],
            "grid_starts": fit["n_starts"],
            "huber_delta": HUBER_DELTA,
            "bootstrap": bootstrap_summary,
            "bootstrap_draws": bootstrap["n_bootstrap"],
            "bootstrap_successes": bootstrap["n_success"],
            "bootstrap_seed": bootstrap["seed"],
            "reference": REFERENCE_PARAMS,
        },
        "verification": {
            "1": {
                "params": {
                    name: float(params[name]) for name in PARAMETER_NAMES
                },
                "checks": verify1_checks,
                "passed": verify1_passed,
            },
            "2": {
                "our_mean_residual": our_mean_residual,
                "published_mean_residual": published_mean_residual,
                "residual_definition": "log(L_pred) - log(L_observed)",
                "published_params": PUBLISHED_PARAMS,
                "checks": verify2_checks,
                "passed": verify2_passed,
            },
            "3": {
                "compute": CHINCHILLA_COMPUTE,
                "our": our_ratio,
                "published": published_ratio,
                "bootstrap_ratio_ci95": [
                    float(ratio_ci[0]),
                    float(ratio_ci[1]),
                ],
                "chinchilla": {
                    "N": CHINCHILLA_N,
                    "D": CHINCHILLA_D,
                    "ratio": CHINCHILLA_D / CHINCHILLA_N,
                },
                "relative_difference_from_chinchilla": chinchilla_difference,
                "checks": verify3_checks,
                "passed": verify3_passed,
            },
        },
        "acceptance": {
            "verify1": verify1_passed,
            "verify2": verify2_passed,
            "verify3": verify3_passed,
            "all_passed": bool(
                verify1_passed and verify2_passed and verify3_passed
            ),
        },
        "figures": {
            "figure19": str(figure19_path),
            "figure20": str(figure20_path),
            "figure21": str(figure21_path),
        },
        "artifacts": {
            "bootstrap": str(bootstrap_path),
            "compute_optimal_csv": str(compute_csv_path),
        },
    }
    results_path = ARTIFACT_DIR / "part6_results.json"
    results_path.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    settings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "matplotlib")
        },
        "data": {
            "url": DATA_URL,
            "file": DATA_FILE.name,
            "columns": {
                "N": "Model Size",
                "C": "Training FLOP",
                "L": "loss",
                "D": "C / (6 N)",
            },
            "remove": "5 highest-loss runs",
        },
        "fit": {
            "loss": f"mean Huber(log L, delta={HUBER_DELTA})",
            "huber_delta": HUBER_DELTA,
            "optimizer": "L-BFGS-B",
            "bounds": {
                "E": [0.1, 5.0],
                "logA": [-10.0, 20.0],
                "logB": [-10.0, 30.0],
                "alpha": [0.05, 1.5],
                "beta": [0.05, 1.5],
            },
            "grid_starts": (
                "E in {1.6,1.8}; A in {200,500,1200}; "
                "B in {800,2200,6000}; alpha,beta in {0.3,0.4}"
            ),
            "bootstrap": {
                "n_resamples": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "starts_per_resample": bootstrap["starts_per_bootstrap"],
                "interval": "2.5th-97.5th percentile",
            },
        },
        "published_parameters": PUBLISHED_PARAMS,
        "reference_parameters": REFERENCE_PARAMS,
        "chinchilla": {
            "compute": CHINCHILLA_COMPUTE,
            "N": CHINCHILLA_N,
            "D": CHINCHILLA_D,
        },
    }
    settings_path = ARTIFACT_DIR / "part6_settings.json"
    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------ report
    lines = [
        "AMAT5315 Week 8 Part 6 - scaling laws",
        "=" * 84,
        "",
        "Data",
        "-" * 84,
        f"  source: {DATA_URL}",
        f"  runs: {data['n_total']} total, 5 highest-loss removed, "
        f"{data['n_used']} used",
        f"  N range: {N.min():.4g} .. {N.max():.4g}",
        f"  D=C/(6N) range: {D.min():.4g} .. {D.max():.4g} tokens",
        f"  loss range: {L.min():.4f} .. {L.max():.4f}",
        "",
        "Fitted Chinchilla law",
        "-" * 84,
        f"  L = E + A/N^alpha + B/D^beta",
        f"  Huber(log L, delta={HUBER_DELTA}) objective: {fit['objective']:.6e}",
        f"  L-BFGS grid starts: {fit['n_starts']}; success: {fit['success']}",
        f"  {'parameter':<8} {'fit':>14} {'std.error':>14} "
        f"{'95% CI lower':>16} {'95% CI upper':>16}",
    ]
    for name in PARAMETER_NAMES:
        summary = bootstrap_summary[name]
        lines.append(
            f"  {name:<8} {params[name]:>14.6f} "
            f"{summary['std_error']:>14.6f} "
            f"{summary['ci95_lower']:>16.6f} "
            f"{summary['ci95_upper']:>16.6f}"
        )
    lines.append(
        f"  reference: E={REFERENCE_PARAMS['E']}, A={REFERENCE_PARAMS['A']}, "
        f"B={REFERENCE_PARAMS['B']}, alpha={REFERENCE_PARAMS['alpha']}, "
        f"beta={REFERENCE_PARAMS['beta']}"
    )
    lines.append("")
    lines.append("VERIFY (1) - parameter ranges")
    lines.append("-" * 84)
    for name, value in verify1_checks.items():
        lines.append(f"  {name}: {value}")
    lines.append(f"  VERIFY (1): {'PASS' if verify1_passed else 'FAIL'}")
    lines.append("")
    lines.append("VERIFY (2) - log-loss residuals (predicted - observed)")
    lines.append("-" * 84)
    lines.append(f"  our mean residual: {our_mean_residual:+.6f}")
    lines.append(
        f"  published mean residual: {published_mean_residual:+.6f}"
    )
    lines.append(
        f"  our RMS residual: {np.sqrt(np.mean(our_residuals**2)):.6f}"
    )
    lines.append(
        f"  published RMS residual: "
        f"{np.sqrt(np.mean(published_residuals**2)):.6f}"
    )
    for name, value in verify2_checks.items():
        lines.append(f"  {name}: {value}")
    lines.append(f"  VERIFY (2): {'PASS' if verify2_passed else 'FAIL'}")
    lines.append("")
    lines.append("VERIFY (3) - compute-optimal allocation")
    lines.append("-" * 84)
    lines.append(
        f"  at C={CHINCHILLA_COMPUTE:.3e}: "
        f"N*={our_ratio['N_star']:.4g}, D*={our_ratio['D_star']:.4g}, "
        f"D*/N*={our_ratio['D_over_N']:.4f}"
    )
    lines.append(
        f"  bootstrap 95% interval for D*/N*: "
        f"[{ratio_ci[0]:.4f}, {ratio_ci[1]:.4f}]"
    )
    lines.append(
        f"  published law at same C: N*={published_ratio['N_star']:.4g}, "
        f"D*={published_ratio['D_star']:.4g}, "
        f"D*/N*={published_ratio['D_over_N']:.4f}"
    )
    lines.append(
        f"  Chinchilla: N={CHINCHILLA_N:.4g}, D={CHINCHILLA_D:.4g}, "
        f"D/N={CHINCHILLA_D / CHINCHILLA_N:.1f}"
    )
    lines.append(
        f"  relative difference from Chinchilla ratio: "
        f"{chinchilla_difference:+.2%}"
    )
    for name, value in verify3_checks.items():
        lines.append(f"  {name}: {value}")
    lines.append(f"  VERIFY (3): {'PASS' if verify3_passed else 'FAIL'}")
    lines.append("")
    lines.append("Outputs")
    lines.append("-" * 84)
    lines.append(f"  Figure 19: {figure19_path}")
    lines.append(f"  Figure 20: {figure20_path}")
    lines.append(f"  Figure 21: {figure21_path}")
    lines.append(f"  bootstrap draws: {bootstrap_path}")
    lines.append(f"  compute-optimal table: {compute_csv_path}")
    lines.append(f"  results: {results_path}")
    lines.append(f"  settings: {settings_path}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 84)
    for number in (1, 2, 3):
        passed = results["acceptance"][f"verify{number}"]
        lines.append(f"  VERIFY ({number}): {'PASS' if passed else 'FAIL'}")

    report = "\n".join(lines) + "\n"
    report_path = EVIDENCE_DIR / "part6_verify.txt"
    report_path.write_text(report, encoding="utf-8")
    print()
    print(report, end="")
    return results


def _params_tuple_from_draw(draw: np.ndarray) -> tuple:
    return tuple(float(value) for value in draw)


if __name__ == "__main__":
    run()
