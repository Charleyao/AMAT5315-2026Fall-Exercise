"""Pooled Pearson chi-squared test with rare-category grouping."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def pooled_chi_square(
    observed: np.ndarray,
    expected: np.ndarray,
    min_expected: float = 5.0,
) -> dict:
    """Pool chi-squared statistics over rows, grouping rare cells per row.

    For every conditioning context (row), next-token categories with expected
    count below ``min_expected`` are merged into one "other" category.  The
    Pearson statistic is then summed across rows and the degrees of freedom are
    the sum of ``(categories_after_grouping - 1)``.
    """
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if observed.shape != expected.shape or observed.ndim != 2:
        raise ValueError("observed and expected must be equal-shaped 2-D arrays")
    if np.any(expected <= 0.0):
        raise ValueError("all expected counts must be positive")

    statistic = 0.0
    degrees_of_freedom = 0
    row_details = []

    for row in range(observed.shape[0]):
        row_expected = expected[row]
        row_observed = observed[row]
        keep = row_expected >= min_expected

        grouped_expected = list(row_expected[keep])
        grouped_observed = list(row_observed[keep])
        if np.any(~keep):
            grouped_expected.append(float(row_expected[~keep].sum()))
            grouped_observed.append(float(row_observed[~keep].sum()))

        grouped_expected_array = np.asarray(grouped_expected, dtype=np.float64)
        grouped_observed_array = np.asarray(grouped_observed, dtype=np.float64)
        if len(grouped_expected_array) < 2:
            raise ValueError(
                f"row {row} has fewer than two categories after grouping"
            )

        row_statistic = float(
            np.sum(
                (grouped_observed_array - grouped_expected_array) ** 2
                / grouped_expected_array
            )
        )
        row_dof = len(grouped_expected_array) - 1
        statistic += row_statistic
        degrees_of_freedom += row_dof
        row_details.append(
            {
                "row": row,
                "observations": int(row_observed.sum()),
                "categories_before": int(observed.shape[1]),
                "categories_after": int(len(grouped_expected_array)),
                "pooled_other": bool(np.any(~keep)),
                "statistic": row_statistic,
                "dof": row_dof,
                "min_expected_kept": float(row_expected[keep].min()),
            }
        )

    p_value = float(chi2.sf(statistic, degrees_of_freedom))
    return {
        "statistic": float(statistic),
        "dof": int(degrees_of_freedom),
        "p_value": p_value,
        "min_expected": float(min_expected),
        "row_details": row_details,
        "total_observations": int(observed.sum()),
        "all_individual_expected_ge_min": all(
            detail["min_expected_kept"] >= min_expected for detail in row_details
        ),
    }
