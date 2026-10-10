#!/usr/bin/env python3
"""Week 5 Part 1 Task 1: manual node-by-node AD of the Lennard-Jones energy.

Differentiate the Lennard-Jones pair energy

    U(r) = 4 * (r**-12 - r**-6)

at r = 1.3 using JAX with 64-bit floats. The computation is broken into the
nodes named by the learning sheet:

    a = r**-6
    b = a**2
    c = b - a
    U = 4*c

Forward mode walks the chain with one ``jax.jvp`` per node, starting from the
tangent dr = 1. Reverse mode walks it backwards with one ``jax.vjp`` per node,
starting from the adjoint dU = 1, and adds the two reverse contributions that
reach the shared node ``a`` (through ``b`` and directly through ``c``).

The result is written to ``week5/artifacts/ad/derivatives.json`` and checked
against the instructor's reference values with absolute tolerance 1e-12.

Run from anywhere with the week-5 JAX environment, e.g.

    week5/.venv/bin/python week5/scripts/lj_ad.py
"""

# 64-bit floats must be enabled before any JAX array is created.
import jax

jax.config.update("jax_enable_x64", True)

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp

WEEK5 = Path(__file__).resolve().parents[1]
DEFAULT_OUT = WEEK5 / "artifacts" / "ad" / "derivatives.json"

R_VALUE = 1.3
TOLERANCE = 1e-12

# Reference values published with the learning sheet.
REFERENCE = {
    "energy": -0.6570169144600471,
    "tangents.U": 2.239979929791143,
    "adjoints.r": 2.239979929791143,
    "jax_grad": 2.239979929791143,
    "adjoints.a": -2.3425903117359734,
}


def energy_nodes(r):
    """The four named nodes, evaluated in order."""
    a = r**-6
    b = a**2
    c = b - a
    U = 4 * c
    return {"a": a, "b": b, "c": c, "U": U}


def forward_mode(r):
    """Chain jax.jvp through one node at a time, tangent(r) = 1."""
    tangents = {}
    dr = jnp.asarray(1.0, dtype=jnp.float64)
    tangents["r"] = dr

    a, da = jax.jvp(lambda x: x**-6, (r,), (dr,))
    tangents["a"] = da

    b, db = jax.jvp(lambda x: x**2, (a,), (da,))
    tangents["b"] = db

    c, dc = jax.jvp(lambda x, y: x - y, (b, a), (db, da))
    tangents["c"] = dc

    U, dU = jax.jvp(lambda x: 4.0 * x, (c,), (dc,))
    tangents["U"] = dU

    values = {"a": a, "b": b, "c": c, "U": U}
    return values, tangents


def reverse_mode(r):
    """Chain jax.vjp backwards, adjoint(U) = 1, summing both paths into a."""
    values = energy_nodes(r)
    a, b, c, U = values["a"], values["b"], values["c"], values["U"]

    adjoints = {"U": jnp.asarray(1.0, dtype=jnp.float64)}

    # U = 4*c
    c_bar = jax.vjp(lambda x: 4.0 * x, c)[1](adjoints["U"])[0]
    adjoints["c"] = c_bar

    # c = b - a gives one contribution to a (the direct path).
    b_bar, a_bar_direct = jax.vjp(lambda x, y: x - y, b, a)[1](c_bar)
    adjoints["b"] = b_bar

    # b = a**2 gives the second contribution to the shared node a.
    (a_bar_square,) = jax.vjp(lambda x: x**2, a)[1](b_bar)

    # Accumulate both reverse contributions to a.
    a_bar = a_bar_direct + a_bar_square
    adjoints["a"] = a_bar

    # a = r**-6
    (r_bar,) = jax.vjp(lambda x: x**-6, r)[1](a_bar)
    adjoints["r"] = r_bar

    return values, adjoints


def jax_gradient(r):
    """Derivative from JAX's built-in reverse-mode gradient."""

    def scalar_energy(x):
        return energy_nodes(x)["U"]

    return jax.grad(scalar_energy)(r)


def analytic_derivative(r):
    """dU/dr = 24 * (r**-7 - 2*r**-13)."""
    return 24.0 * (r**-7 - 2.0 * r**-13)


def as_float(value):
    return float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    r = jnp.asarray(R_VALUE, dtype=jnp.float64)

    forward_values, tangents = forward_mode(r)
    reverse_values, adjoints = reverse_mode(r)

    # Forward and reverse must visit the same node values.
    for name in ("a", "b", "c", "U"):
        assert jnp.array_equal(forward_values[name], reverse_values[name]), name

    gradient = jax_gradient(r)
    analytic = analytic_derivative(R_VALUE)

    result = {
        "r": as_float(r),
        "energy": as_float(forward_values["U"]),
        "tangents": {
            "r": as_float(tangents["r"]),
            "a": as_float(tangents["a"]),
            "b": as_float(tangents["b"]),
            "c": as_float(tangents["c"]),
            "U": as_float(tangents["U"]),
        },
        "adjoints": {
            "r": as_float(adjoints["r"]),
            "a": as_float(adjoints["a"]),
            "b": as_float(adjoints["b"]),
            "c": as_float(adjoints["c"]),
            "U": as_float(adjoints["U"]),
        },
        "jax_grad": as_float(gradient),
    }

    measured = {
        "energy": result["energy"],
        "tangents.U": result["tangents"]["U"],
        "adjoints.r": result["adjoints"]["r"],
        "jax_grad": result["jax_grad"],
        "adjoints.a": result["adjoints"]["a"],
    }

    print(f"r = {result['r']!r}")
    print(f"floating-point dtype = {r.dtype}")
    print("\nnode values:")
    for name in ("a", "b", "c", "U"):
        print(f"  {name:>2} = {as_float(forward_values[name]):.16f}")
    print("\nderivative checks against reference (tolerance 1e-12):")
    failures = []
    for key, expected in REFERENCE.items():
        actual = measured[key]
        error = abs(actual - expected)
        status = "PASS" if error < TOLERANCE else "FAIL"
        if status == "FAIL":
            failures.append(key)
        print(f"  {key:>12}: measured={actual:.16f} expected={expected:.16f} "
              f"abs_err={error:.3e} {status}")
    print("\nderivative agreement with 24*(r**-7 - 2*r**-13):")
    for key in ("tangents.U", "adjoints.r", "jax_grad"):
        error = abs(measured[key] - analytic)
        status = "PASS" if error < TOLERANCE else "FAIL"
        if status == "FAIL":
            failures.append(key + "-analytic")
        print(f"  {key:>12}: abs_err={error:.3e} {status}")
    print(f"  analytic      = {analytic:.16f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {args.out}")

    if failures:
        print("FAILED:", ", ".join(failures), file=sys.stderr)
        return 1
    print("all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
