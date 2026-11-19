"""EXPERIMENT 3, fair baseline: adds ONE controller to run_tracking.py's results.

  local_linear  upright LQR gain + feedforward from the SAME linear Taylor
                model, computed exactly like the Koopman trackers' feedforward
                (one-step least squares).  It differs from the Koopman trackers
                only in the model: linear (Taylor) vs lifted (Koopman).

local_fixed and tvlqr (run_tracking.py) get the EXACT nonlinear model for
their feedforward, so they are a ceiling, not a peer.  This script simulates
only local_linear, under the same scenes, letters and conditions, and compares
it with the controllers run_tracking.py already saved.  Nothing is rerun and
run_tracking.py's files are not changed.

    python run_tracking_fair.py               # every scene in TRACK_RUN
    python run_tracking_fair.py --scenes small
    python run_tracking_fair.py --figures     # only redraw from saved results
Needs: python run_tracking.py (same config.py) first.

Writes, next to run_tracking.py's results:
    results/tracking/<tag>/<scene>/<letter>/results_linear.csv, paths_linear.npz
    results/tracking/<tag>/summary_fair.md
    figures/tracking_fair_<scene>_<letter>.png
"""
import argparse
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
import tool_lib as tl
from tool_lib import tracking_metrics
from control import design_local_lqr, linear_controller, path_stability
from run_tracking import (Scene, tracking_conditions, tracking, simulate, _load,
                          _workspace, _limits, FIG, PHYSICS)

NAME = "local_linear"
# Style of the new baseline (works with any tool_lib.py version)
for _d, _v in ((tl.COLOR, "#2a78d6"), (tl.LS, ":"), (tl.MARKER, "P"),
               (tl.LABEL, "Local LQR + linear feedforward (fair baseline)"),
               (tl.SHORT, "Local LQR (linear)")):
    _d.setdefault(NAME, _v)


# ------------------------------------------------------------ simulation
def run_letter(sc, letter):
    out = sc.out_dir(letter)
    if not os.path.exists(os.path.join(out, "results.csv")):
        raise SystemExit(f"{out}/results.csv missing: run python run_tracking.py first")
    pts = sc.points(letter)
    rows, arrays = [], {}
    print(f"\n=== Fair baseline, {sc.what(letter)} ===")
    for cond in tracking_conditions():
        with tracking(cond, sc):
            ref = sc.reference(letter)
            ctrl = linear_controller(ref, design_local_lqr())   # model of this condition
            unst, rho = path_stability(ctrl, ref)
            T, X, U = simulate(ctrl, ref)
            m, P = tracking_metrics(T, X, U, ref, pts)
            rows.append(dict(scene=sc.name, letter=letter, condition=cond.name,
                             group=cond.group, label=cond.label(), controller=NAME,
                             path_unstable_pct=unst, path_max_rho=rho, **m))
            n = min(len(X), len(U), len(ref["T"]))
            arrays[f"path|{cond.name}|{NAME}"] = P
            arrays[f"err|{cond.name}|{NAME}"] = np.linalg.norm(P[:n] - ref["P"][:n], axis=1)
            arrays[f"u|{cond.name}|{NAME}"] = U[:n]
            print(f"  {cond.label():30s} shape {m['shape_mm']:8.2f} mm  "
                  f"RMSE {m['rmse_mm']:8.2f} mm  sat {m['sat_pct']:5.1f}%  "
                  f"gain unstable on {unst:5.1f}% of path"
                  f"{'  DIVERGED' if m['diverged'] else ''}", flush=True)
    with open(os.path.join(out, "results_linear.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez(os.path.join(out, "paths_linear.npz"), **arrays)


# ------------------------------------------------------------ comparison
def _load_all(sc, letter):
    """run_tracking.py's controllers + local_linear, as one set of rows/arrays."""
    rows, arr, order = _load(sc, letter)
    out = sc.out_dir(letter)
    lin = list(csv.DictReader(open(os.path.join(out, "results_linear.csv"))))
    for r in lin:
        for k in ("rmse_mm", "max_mm", "shape_mm", "final_mm", "sat_pct", "peak_u1",
                  "peak_u2", "path_unstable_pct", "path_max_rho"):
            r[k] = float(r[k])
        r["diverged"] = r["diverged"] == "True"
    rows = [r for r in rows if r["controller"] != NAME] + lin
    arr.update(dict(np.load(os.path.join(out, "paths_linear.npz"))))
    koop = [n for n in order if n not in PHYSICS and n != NAME]
    return rows, arr, koop


def _get(rows, cond_name, n):
    r = [q for q in rows if q["condition"] == cond_name and q["controller"] == n]
    return r[0] if r else None


def _cell(r):
    if r is None:
        return "—"
    return "diverged" if r["diverged"] else f"{r['shape_mm']:.2f}"


def fig_fair(sc, letter, rows, arr, koop):
    """Target letter, local_linear vs every Koopman tracker (the fair pair);
    the exact-model controllers drawn thin and grey as the ceiling."""
    conds = tracking_conditions()
    pts = arr["letter"]
    fig, ax = plt.subplots(1, len(conds), figsize=(3.8 * len(conds), 4.2), squeeze=False)
    for a, cond in zip(ax[0], conds):
        _workspace(a)
        a.plot(pts[:, 0], pts[:, 1], color=tl.NEUTRAL, lw=6, solid_capstyle="round",
               label="target letter")
        for n in PHYSICS:
            P = arr.get(f"path|{cond.name}|{n}")
            if P is not None:
                a.plot(P[:, 0], P[:, 1], color=tl.TEXT2, lw=0.8, alpha=0.6,
                       label="exact-model controllers (ceiling)" if n == PHYSICS[0] else None)
        for n in [NAME] + koop:
            P = arr.get(f"path|{cond.name}|{n}")
            if P is not None:
                a.plot(P[:, 0], P[:, 1], color=tl.COLOR[n], ls=tl.LS[n], lw=1.8,
                       label=tl.LABEL[n])
        _limits(a, sc, pts)
        a.set_title(cond.label(), fontsize=9)
        a.set_xlabel("x (m)")
    ax[0, 0].set_ylabel("y (m)")
    fig.tight_layout()
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8)
    tl.title(fig, "Experiment 3, fair comparison: linear model vs Koopman model, same "
                  "controller structure (LQR gain + least-squares feedforward)",
             f"{sc.what(letter)}; torque limit {sc.torque_limit} N·m; grey = controllers "
             "given the exact nonlinear model (upper bound)", y=1.1)
    fig.savefig(f"{FIG}/tracking_fair_{sc.tag(letter)}.png"); plt.close(fig)


def table(scenes):
    """Shape error (mm): local_linear vs Koopman (the fair pair), then the ceiling."""
    md = ["Experiment 3, fair comparison. Shape error in mm (mean distance to the "
          "letter; lower is better). `Local LQR (linear)` and the Koopman trackers use "
          "the same controller structure and differ only in the model; the last two "
          "columns are given the exact nonlinear model (upper bound).", ""]
    header = None
    for sc in scenes:
        for letter in sc.letters:
            out = sc.out_dir(letter)
            if not os.path.exists(os.path.join(out, "results_linear.csv")):
                continue
            rows, arr, koop = _load_all(sc, letter)
            fig_fair(sc, letter, rows, arr, koop)
            cols = [NAME] + koop + PHYSICS
            if header is None:
                header = cols
                md += ["| scene, letter | condition | " + " | ".join(tl.SHORT[n] for n in cols)
                       + " | best Koopman vs linear |",
                       "|" + "---|" * (len(cols) + 3)]
            for cond in tracking_conditions():
                lin = _get(rows, cond.name, NAME)
                ks = [_get(rows, cond.name, n) for n in koop]
                ok = [(r["shape_mm"], r["controller"]) for r in ks if r and not r["diverged"]]
                if not ok or lin is None:
                    verdict = "—"
                elif lin["diverged"]:
                    verdict = f"Koopman ({tl.SHORT[min(ok)[1]]}); linear diverged"
                else:
                    best, who = min(ok)
                    ratio = best / lin["shape_mm"]
                    verdict = (f"Koopman better ({tl.SHORT[who]}, ×{ratio:.2f})" if ratio < 0.99
                               else f"linear better (best Koopman ×{ratio:.2f})" if ratio > 1.01
                               else "tie")
                md.append(f"| {sc.name}, {letter} | {cond.label()} | "
                          + " | ".join(_cell(_get(rows, cond.name, n)) for n in cols)
                          + f" | {verdict} |")
    if header is None:
        print("no results_linear.csv found: run without --figures first")
        return
    md += ["", "×r = best Koopman shape error / local_linear shape error "
               "(below 1: Koopman better; within 1%: tie)."]
    path = os.path.join("results", "tracking", tl.TAG, "summary_fair.md")
    open(path, "w").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md) + f"\n\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=list(cfg.TRACK_RUN),
                    choices=list(cfg.TRACK_SCENES))
    ap.add_argument("--figures", action="store_true", help="only redraw from saved results")
    a = ap.parse_args()
    scenes = [Scene(s) for s in a.scenes]
    if not a.figures:
        for sc in scenes:
            for letter in sc.letters:
                run_letter(sc, letter)
    table(scenes)


if __name__ == "__main__":
    main()