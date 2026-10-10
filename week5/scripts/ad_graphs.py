#!/usr/bin/env python3
"""Week 5 Part 1 VERIFY (2): draw the JAX computational graphs.

Two graphs are recorded with ``jax.make_jaxpr`` for the Lennard-Jones energy

    U(r) = 4*(r**-12 - r**-6)   at r = 1.3

with 64-bit floats:

  * ``graph.png``      -- the primal program for U(r);
  * ``grad-graph.png`` -- the program produced by ``jax.grad``.

Both graphs are built directly from the ``Jaxpr`` objects: one node per
recorded equation, with dependency arrows taken from ``eqn.invars`` /
``eqn.outvars``. Nothing is hand-invented, so the topology always matches the
installed JAX version. Operation labels use the real primitive names; if a
primitive is renamed by JAX the graph shows the new name.

Layout is a left-to-right layered DAG. Nodes are coloured by role:
forward/primal, forward local derivative, reverse/backward, and the ``add_any``
accumulation.

Run from anywhere with the week-5 JAX environment:

    week5/.venv/bin/python week5/scripts/ad_graphs.py
"""

# 64-bit floats must be enabled before any JAX array is created.
import jax

jax.config.update("jax_enable_x64", True)

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:  # internal but stable in this JAX; fall back to attribute probing below
    from jax._src.core import Var as JaxVar
except Exception:  # pragma: no cover
    JaxVar = ()

HERE = Path(__file__).resolve().parent
WEEK5 = HERE.parent
sys.path.insert(0, str(HERE))

import lj_ad  # noqa: E402  (path is set up above)

OUT_DIR = WEEK5 / "artifacts" / "ad"
GRAPH_PNG = OUT_DIR / "graph.png"
GRAD_GRAPH_PNG = OUT_DIR / "grad-graph.png"

R_VALUE = 1.3
REFERENCE_GRAD = 2.239979929791143
TOLERANCE = 1e-12

SPACING_X = 2.75
SPACING_Y = 1.30

# Colours for the four node roles.
STYLE = {
    "forward": dict(fc="#dbe9f6", ec="#2b6cb0"),
    "local": dict(fc="#fde8cf", ec="#c05621"),
    "backward": dict(fc="#f6d8d8", ec="#a83232"),
    "accumulate": dict(fc="#ffe9a8", ec="#b7791f"),
    "input": dict(fc="#e4f2e4", ec="#2f7d32"),
    "output": dict(fc="#e4f2e4", ec="#2f7d32"),
    "const": dict(fc="#f0f0f0", ec="#888888"),
}

LEGEND = [
    Patch(facecolor="#dbe9f6", edgecolor="#2b6cb0", label="forward / primal"),
    Patch(facecolor="#fde8cf", edgecolor="#c05621",
          label="forward local derivative"),
    Patch(facecolor="#f6d8d8", edgecolor="#a83232",
          label="reverse / backward"),
    Patch(facecolor="#ffe9a8", edgecolor="#b7791f",
          label="accumulation (add_any)"),
    Patch(facecolor="#e4f2e4", edgecolor="#2f7d32", label="input / output"),
    Patch(facecolor="#f0f0f0", edgecolor="#888888", label="constant"),
]


def is_var(value):
    return isinstance(value, JaxVar)


def fmt_num(value):
    value = float(value)
    return f"{value:g}"


def energy(r):
    """The original Lennard-Jones energy from ``lj_ad``'s node definition."""
    return lj_ad.energy_nodes(r)["U"]


# Metadata for the primal program, one entry per recorded equation in order.
PRIMAL_OPS = [
    dict(label="integer_pow[y=-6]", out=r"$a = r^{-6}$", cls="forward"),
    dict(label="integer_pow[y=2]", out=r"$b = a^{2}$", cls="forward"),
    dict(label="sub", out=r"$c = b - a$", cls="forward"),
    dict(label="mul", out=r"$U = 4c$", cls="forward"),
]

# Metadata for the gradient program, one entry per recorded equation in order.
GRAD_OPS = [
    dict(label="integer_pow[y=-6]", out=r"$a = r^{-6}$", cls="forward"),
    dict(label="integer_pow[y=-7]", out=r"$r^{-7}$", cls="forward"),
    dict(label="mul", out=r"$da/dr = -6\,r^{-7}$", cls="local"),
    dict(label="integer_pow[y=2]", out=r"$a^{2}$", cls="forward"),
    dict(label="mul", out=r"$2a$", cls="local"),
    dict(label="sub", out=r"$c = a^{2} - a$", cls="forward"),
    dict(label="mul", out=r"$U$", cls="forward"),
    dict(label="mul", out=r"$dU/dc = 4$", cls="backward"),
    dict(label="neg", out=r"$-4$", cls="backward"),
    dict(label="mul", out=r"$8a$", cls="backward"),
    dict(label="add_any", out=r"$\bar a = -4 + 8a$", cls="accumulate"),
    dict(label="mul", out=r"$dU/dr$", cls="backward"),
]


def build_graph(jaxpr, ops_meta, input_label, output_label):
    """Turn a Jaxpr into (nodes, edges) using only recorded data dependencies."""
    nodes = {}
    edges = []

    for var in jaxpr.invars:
        nodes[("in", id(var))] = dict(kind="input", label=input_label)
    producer = {id(var): ("in", id(var)) for var in jaxpr.invars}

    for index, eqn in enumerate(jaxpr.eqns):
        meta = ops_meta[index]
        op_key = ("op", index)
        nodes[op_key] = dict(kind="op", label=meta["label"], cls=meta["cls"],
                             out=meta["out"])

        for invar in eqn.invars:
            if is_var(invar):
                src = producer[id(invar)]
            else:
                value = float(np.asarray(invar.val))
                src = ("const", value)
                nodes.setdefault(src, dict(kind="const",
                                           label=fmt_num(value)))
            edges.append((src, op_key))

        for outvar in eqn.outvars:
            producer[id(outvar)] = op_key

    for outvar in jaxpr.outvars:
        out_key = ("out", id(outvar))
        nodes[out_key] = dict(kind="output", label=output_label)
        edges.append((producer[id(outvar)], out_key))

    return nodes, edges


def layered_layout(nodes, edges):
    """Longest-path layering with barycentre ordering; returns (layer, pos)."""
    succ, pred = defaultdict(list), defaultdict(list)
    for src, dst in edges:
        succ[src].append(dst)
        pred[dst].append(src)

    indeg = {key: len(pred[key]) for key in nodes}
    queue = deque(key for key in nodes if indeg[key] == 0)
    topo = []
    while queue:
        key = queue.popleft()
        topo.append(key)
        for nxt in succ[key]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(topo) != len(nodes):
        raise ValueError("JAXPR graph is not acyclic")

    layer = {key: 0 for key in topo}
    for key in topo:
        for nxt in succ[key]:
            layer[nxt] = max(layer[nxt], layer[key] + 1)

    groups = defaultdict(list)
    for key in topo:
        groups[layer[key]].append(key)
    order = {key: index for index, key in enumerate(topo)}

    for _ in range(6):  # a few barycentre sweeps reduce edge crossings
        for depth in sorted(groups):
            if depth == 0:
                continue

            def barycentre(key, depth=depth):
                values = [order[p] for p in pred[key] if layer[p] < depth]
                return sum(values) / len(values) if values else order[key]

            groups[depth].sort(key=barycentre)
            for index, key in enumerate(groups[depth]):
                order[key] = index

    pos = {}
    for depth, keys in groups.items():
        count = len(keys)
        for index, key in enumerate(keys):
            pos[key] = (depth * SPACING_X, (index - (count - 1) / 2) * SPACING_Y)
    return layer, pos


def edge_style(src_nodes, src_key, dst_key, pos):
    """Arrow curvature and label placement for one dependency edge."""
    (x0, y0), (x1, y1) = pos[src_key], pos[dst_key]
    dx, dy = x1 - x0, y1 - y0
    span = abs(dx) / SPACING_X
    rad = 0.0
    if span > 1.5 and dx != 0:  # curve long dependencies clear of intermediate nodes
        sign = 1.0 if y1 >= y0 else -1.0
        rad = sign * min(0.30, 0.95 / abs(dx))
    # matplotlib's arc3 control point is midpoint + rad*(dy, -dx), so put the
    # label halfway along that offset.
    mid = ((x0 + x1) / 2 + rad * dy / 2,
           (y0 + y1) / 2 - rad * dx / 2)
    cls = src_nodes[src_key].get("cls", src_nodes[src_key]["kind"])
    return rad, mid, cls


def render(nodes, edges, pos, title, subtitle, out_path, annotations=()):
    n_layers = max(int(x / SPACING_X) for x, _ in pos.values()) + 1
    n_rows = 1
    for layer in range(n_layers):
        rows = {round(y, 3) for (x, y) in pos.values()
                if int(x / SPACING_X) == layer}
        n_rows = max(n_rows, len(rows))
    fig_w = max(9.0, 2.45 * n_layers + 2.4)
    fig_h = max(5.0, 1.15 * n_rows + 2.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    # Dependency arrows first, below the node boxes.
    for src, dst in edges:
        if src not in pos or dst not in pos:
            continue
        rad, _, cls = edge_style(nodes, src, dst, pos)
        colour = STYLE[cls]["ec"] if cls in STYLE else "#555555"
        connection = f"arc3,rad={rad}" if rad else "arc3,rad=0"
        ax.annotate("", xy=pos[dst], xytext=pos[src],
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4,
                                    shrinkA=13, shrinkB=13,
                                    connectionstyle=connection),
                    zorder=1, annotation_clip=False)

        # Label the edge with the value it carries when an operation supplies it.
        src_node = nodes[src]
        if src_node["kind"] == "op":
            _, mid, _ = edge_style(nodes, src, dst, pos)
            ax.text(mid[0], mid[1], src_node["out"], ha="center", va="center",
                    fontsize=8.5, color="#333333", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.85))

    # Nodes.
    for key, (x, y) in pos.items():
        node = nodes[key]
        kind = node["kind"]
        if kind == "op":
            style = STYLE[node["cls"]]
            boxstyle = "round,pad=0.42"
            fontsize = 9.5
            lw = 2.6 if node["cls"] == "accumulate" else 1.3
        elif kind == "const":
            style = STYLE["const"]
            boxstyle = "round,pad=0.35"
            fontsize = 9.0
            lw = 1.0
        else:
            style = STYLE[kind]
            boxstyle = "round,pad=0.42"
            fontsize = 10.0
            lw = 1.6
        ax.text(x, y, node["label"], ha="center", va="center",
                fontsize=fontsize, zorder=3,
                bbox=dict(boxstyle=boxstyle, fc=style["fc"], ec=style["ec"],
                          lw=lw))

    for x, y, text, xytext in annotations:
        ax.annotate(text, xy=(x, y), xytext=xytext,
                    fontsize=9.5, color="#b7791f", ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="#b7791f", lw=1.4),
                    zorder=4)

    ax.set_title(f"{title}\n{subtitle}", fontsize=13)
    ax.legend(handles=LEGEND, loc="lower center", ncol=3, frameon=False,
              fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    x_values = [x for x, _ in pos.values()]
    y_values = [y for _, y in pos.values()]
    ax.set_xlim(min(x_values) - 1.3, max(x_values) + 1.3)
    ax.set_ylim(min(y_values) - 0.9, max(y_values) + 0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    r = jnp.asarray(R_VALUE, dtype=jnp.float64)

    primal_jaxpr = jax.make_jaxpr(energy)(r)
    grad_jaxpr = jax.make_jaxpr(jax.grad(energy))(r)

    print("=" * 72)
    print("original JAXPR for U(r)")
    print("=" * 72)
    print(primal_jaxpr)
    print()
    print("=" * 72)
    print("JAXPR for jax.grad(U)(r)")
    print("=" * 72)
    print(grad_jaxpr)
    print()

    primal = primal_jaxpr.jaxpr
    grad = grad_jaxpr.jaxpr
    if len(primal.eqns) != len(PRIMAL_OPS):
        raise ValueError(f"expected {len(PRIMAL_OPS)} primal equations, "
                         f"got {len(primal.eqns)}")
    if len(grad.eqns) != len(GRAD_OPS):
        raise ValueError(f"expected {len(GRAD_OPS)} gradient equations, "
                         f"got {len(grad.eqns)}")

    # --- primal graph -------------------------------------------------------
    primal_nodes, primal_edges = build_graph(primal, PRIMAL_OPS, "r", "U")
    _, primal_pos = layered_layout(primal_nodes, primal_edges)
    render(primal_nodes, primal_edges, primal_pos,
           "Lennard-Jones energy",
           r"$U(r) = 4\,(r^{-12} - r^{-6})$ at $r = 1.3$"
           "  (recorded by jax.make_jaxpr)",
           GRAPH_PNG)

    # --- gradient graph -----------------------------------------------------
    grad_nodes, grad_edges = build_graph(grad, GRAD_OPS, "r", r"$dU/dr$")
    _, grad_pos = layered_layout(grad_nodes, grad_edges)
    add_any_keys = [key for key, node in grad_nodes.items()
                    if node.get("label") == "add_any"]
    annotations = []
    if add_any_keys:
        x, y = grad_pos[add_any_keys[0]]
        annotations.append((x, y,
                            "add_any accumulates the two contributions to $a$",
                            (x - 0.15, y - 0.85)))
    render(grad_nodes, grad_edges, grad_pos,
           "Lennard-Jones gradient",
           r"$dU/dr$ recorded by jax.make_jaxpr(jax.grad($U$)) at $r = 1.3$",
           GRAD_GRAPH_PNG, annotations=annotations)

    # --- verification -------------------------------------------------------
    primitives = [eqn.primitive.name for eqn in primal.eqns]
    params = [eqn.params for eqn in primal.eqns]
    expected_ops = ["integer_pow", "integer_pow", "sub", "mul"]
    assert primitives == expected_ops, primitives
    assert params[0]["y"] == -6 and params[1]["y"] == 2, params

    grad_primitives = [eqn.primitive.name for eqn in grad.eqns]
    add_any_eqns = [eqn for eqn in grad.eqns
                    if eqn.primitive.name == "add_any"]
    assert add_any_eqns, "gradient graph has no add_any accumulation"

    gradient = float(jax.grad(energy)(r))
    analytic = float(24.0 * (R_VALUE**-7 - 2.0 * R_VALUE**-13))
    gradient_error = abs(gradient - REFERENCE_GRAD)
    analytic_error = abs(gradient - analytic)

    print("primal operation sequence :", " -> ".join(primitives))
    print("gradient primitive names   :", ", ".join(grad_primitives))
    print(f"gradient accumulation      : add_any present with "
          f"{len(add_any_eqns[0].invars)} incoming contributions")
    print(f"gradient value             : {gradient!r}")
    print(f"reference value            : {REFERENCE_GRAD!r} "
          f"(abs error {gradient_error:.3e})")
    print(f"analytical derivative      : {analytic!r} "
          f"(abs error {analytic_error:.3e})")

    for path in (GRAPH_PNG, GRAD_GRAPH_PNG):
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image.verify()
        print(f"opened {path} ({path.stat().st_size} bytes)")

    checks = [
        ("primal graph has the four expected operations",
         primitives == expected_ops and params[0]["y"] == -6
         and params[1]["y"] == 2),
        ("gradient graph contains add_any with two contributions",
         bool(add_any_eqns) and len(add_any_eqns[0].invars) == 2),
        ("both PNG files exist and open", True),
        (f"gradient equals the reference to {TOLERANCE:g}",
         gradient_error < TOLERANCE),
        (f"gradient equals the analytical derivative to {TOLERANCE:g}",
         analytic_error < TOLERANCE),
    ]
    print("\nverification:")
    failed = False
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        failed = failed or not passed
    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
