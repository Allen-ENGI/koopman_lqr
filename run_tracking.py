"""EXPERIMENT 3: tracking a continuous path (the arm tip draws letters).

Two scenes (config.py, (h)) ask the same question as the balance maps,
"small region or large region?":
  small  a small letter above the shoulder, near upright
  large  letters centred on the shoulder: the arm sweeps the whole workspace

Varied (config.py, section 1):
  observables     one Koopman controller per set in TRACK_OBSERVABLES
  robustness      TRACK_ROBUSTNESS (default: model error, i.e. the physics
                  designs use wrong masses while Koopman learns from the true arm)
Fixed per scene: letters, centre, size, drawing time, torque limit; Q, R, dt.

Controllers (control.py), each using only its own model:
  local_fixed  upright LQR gain + EXACT nonlinear inverse-dynamics feedforward
  tvlqr        time-varying LQR along the path + exact inverse dynamics
               (both get the full nonlinear model: a ceiling, not a peer;
               the fair local-linearization baseline is run_tracking_fair.py)
  Koopman      every set in TRACK_OBSERVABLES, refit on data covering the letter
Physics controllers are designed under each condition (so model error reaches
them); Koopman controllers are fitted once per letter from the true arm's data.

    python run_tracking.py --check            # reachable, within the torque limit?
    python run_tracking.py                    # every scene in TRACK_RUN
    python run_tracking.py --scenes small     # one scene
    python run_tracking.py --figures          # only redraw from saved results
    python run_tracking.py --no-gif           # skip the (slower) GIFs
Needs results/ident_data.npz (python run_koopman.py --stage data).
"""
import argparse
import csv
import os
from contextlib import contextmanager

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

import config as cfg
import tool_lib as tl
from tool_lib import (build_reference, letter_points, max_angle_error, tracking_metrics,
                      letter_size, reach)
from control import (design_local_lqr, physics_controllers, fit_tracking_models,
                     koopman_controllers, path_stability, _wrapped_error)

FIG = "figures"
PHYSICS = ["local_fixed", "tvlqr"]
tl.apply_style()


class Scene:
    """One entry of config.TRACK_SCENES, with sizes and limits resolved."""

    def __init__(self, name):
        s = cfg.TRACK_SCENES[name]
        self.name = name
        self.letters = [str(x).upper() for x in s["letters"]]
        self.center = tuple(float(v) for v in s["center"])
        self.size = letter_size(s["size"])
        self.draw_time = float(s["draw_time"])
        self.torque_limit = [float(v) for v in
                             np.minimum(s["torque_limit"], cfg.MOTOR_TORQUE_LIMIT)]

    def what(self, letter):
        where = ("centred on the shoulder" if np.allclose(self.center, 0) else
                 f"centred at ({self.center[0]:g}, {self.center[1]:g}) m")
        return (f"{self.name} scene: letter '{letter}', {100 * self.size:.0f} cm high, "
                f"{where}, drawn in {self.draw_time:g} s")

    def reference(self, letter):
        return build_reference(letter, self.size, self.draw_time, self.center)

    def points(self, letter):
        return letter_points(letter, self.size, self.center)

    def out_dir(self, letter):
        return os.path.join("results", "tracking", tl.TAG, self.name, letter)

    def tag(self, letter):
        return f"{self.name}_{letter}"


def tracking_conditions():
    """Main condition plus the robustness variables tracking uses."""
    return tl.conditions("all", groups=cfg.TRACK_ROBUSTNESS)


@contextmanager
def tracking(cond, scene):
    """Apply a condition, with the scene's torque limit."""
    with tl.applied(cond):
        cfg.TORQUE_LIMIT = list(scene.torque_limit)
        yield cond


def describe(cond, scene):
    """cond.describe() with the scene's torque limit."""
    return cond.describe().replace(f"torque limit [{tl._fmt(cond.torque_limit)}]",
                                   f"torque limit [{tl._fmt(scene.torque_limit)}]")


def simulate(ctrl, ref):
    """One run on the TRUE plant of the current condition; the same noise
    seed for every controller."""
    return tl.simulate(ctrl, ref["X"][0], tf=ref["T"][-1], seed=cfg.SEED)


# ------------------------------------------------------------ check
def check(scenes):
    """Is each letter reachable, and within the torque limit?"""
    np.set_printoptions(precision=3, suppress=True)
    lo, hi = reach()
    print(f"arm reaches {lo:.2f}-{hi:.2f} m from the shoulder")
    for sc in scenes:
        for letter in sc.letters:
            r = np.hypot(*sc.points(letter).T)               # distance from the shoulder
            print(f"{sc.what(letter)}: tip distance {r.min():.3f}-{r.max():.3f} m; "
                  f"torque limit {sc.torque_limit} N m")
            for cond in tracking_conditions():
                with tracking(cond, sc):
                    ref = sc.reference(letter)
                    d = _wrapped_error(ref["X"])
                    print(f"  {cond.label():22s} torque needed {ref['torque_need']}  "
                          f"{'ok' if ref['feasible'] else 'EXCEEDS LIMIT'}")
            print(f"  joint offsets from upright along the path: "
                  f"dq1 {d[:, 0].min():+.2f}..{d[:, 0].max():+.2f} rad, "
                  f"dq2 {d[:, 1].min():+.2f}..{d[:, 1].max():+.2f} rad "
                  f"(balance Koopman models: |dq| <= {cfg.KOOPMAN_R_TRAIN} rad; tracking "
                  f"refits on |dq| <= {min(np.pi, max_angle_error(ref) + cfg.TRACK_R_MARGIN):.2f} rad)")


# ------------------------------------------------------------ simulation
def run_letter(sc, letter):
    out = sc.out_dir(letter)
    os.makedirs(out, exist_ok=True)
    print(f"\n=== Experiment 3: {sc.what(letter)} ===")
    # Koopman: fitted ONCE from the true arm's data (does not use the physics model)
    with tracking(tl.MAIN, sc):
        ref0 = sc.reference(letter)
        print(f"torque needed on the nominal arm {np.round(ref0['torque_need'], 2)} N·m "
              f"(limit {sc.torque_limit}): "
              f"{'feasible' if ref0['feasible'] else 'EXCEEDS THE LIMIT'}")
        koop = koopman_controllers(ref0, fit_tracking_models(ref0))
    order = PHYSICS + [n for n in cfg.TRACK_OBSERVABLES if n in koop]
    pts = sc.points(letter)
    rows, arrays = [], {"letter": pts, "ref": ref0["P"], "order": np.array(order)}

    for cond in tracking_conditions():
        with tracking(cond, sc):
            ref = sc.reference(letter)              # same path; torque need on this arm
            # physics: designed under this condition (model error reaches them here)
            ctrls = dict(physics_controllers(ref, design_local_lqr()["K"]))
            ctrls.update(koop)
            print(f"\n[{sc.tag(letter)} | {cond.label()}]  ({describe(cond, sc)})")
            for name in order:
                # frozen-time check of the gain along the path (true plant)
                unst, rho = path_stability(ctrls[name], ref)
                T, X, U = simulate(ctrls[name], ref)
                m, P = tracking_metrics(T, X, U, ref, pts)
                rows.append(dict(scene=sc.name, letter=letter, condition=cond.name,
                                 group=cond.group, label=cond.label(), controller=name,
                                 path_unstable_pct=unst, path_max_rho=rho, **m))
                n = min(len(X), len(U), len(ref["T"]))
                arrays[f"path|{cond.name}|{name}"] = P
                arrays[f"err|{cond.name}|{name}"] = np.linalg.norm(P[:n] - ref["P"][:n], axis=1)
                arrays[f"u|{cond.name}|{name}"] = U[:n]
                if cond.group == "main":
                    arrays[f"x|main|{name}"] = np.asarray(X)[:n]    # for the GIF
                print(f"  {tl.LABEL[name]:40s} shape {m['shape_mm']:8.2f} mm  "
                      f"RMSE {m['rmse_mm']:8.2f} mm  sat {m['sat_pct']:5.1f}%  "
                      f"gain unstable on {unst:5.1f}% of path (max radius {rho:.3f})"
                      f"{'  DIVERGED' if m['diverged'] else ''}", flush=True)
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.savez(os.path.join(out, "paths.npz"), **arrays)


# ------------------------------------------------------------ figures
def _load(sc, letter):
    out = sc.out_dir(letter)
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    for r in rows:
        for k in ("rmse_mm", "max_mm", "shape_mm", "final_mm", "sat_pct", "peak_u1", "peak_u2",
                  "path_unstable_pct", "path_max_rho"):
            if k in r:
                r[k] = float(r[k])
        r["diverged"] = r["diverged"] == "True"
    arr = dict(np.load(os.path.join(out, "paths.npz")))
    return rows, arr, [str(n) for n in arr["order"]]


def _cell(rows, cond_name, n):
    r = [q for q in rows if q["condition"] == cond_name and q["controller"] == n]
    if not r:
        return "—"
    return "diverged" if r[0]["diverged"] else f"{r[0]['shape_mm']:.2f}"


def _unstable(rows, n):
    r = [q for q in rows if q["condition"] == "main" and q["controller"] == n]
    if not r or "path_unstable_pct" not in r[0]:
        return "—"
    return f"{r[0]['path_unstable_pct']:.0f}%"


def _workspace(a):
    """Shoulder motor and the ring the tip can reach."""
    for r in reach():                                   # the shoulder is at the origin
        a.add_patch(Circle((0.0, 0.0), r, fill=False, ls=":", lw=0.8, color=tl.TEXT2))
    a.plot([0], [0], marker="o", ms=6, color=tl.TEXT, zorder=5)


def _limits(a, sc, pts):
    if np.allclose(sc.center, 0):                       # whole workspace
        L = 1.1 * reach()[1]
        a.set_xlim(-L, L); a.set_ylim(-L, L)
    else:                                               # zoom on the small letter
        pad = 0.35 * sc.size
        a.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        a.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    a.set_aspect("equal")


def fig_paths(sc, letter, arr, order):
    """One panel per condition: the target letter and every controller's tip path."""
    conds = tracking_conditions()
    pts = arr["letter"]
    fig, ax = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 4.0), squeeze=False)
    for a, cond in zip(ax[0], conds):
        _workspace(a)
        a.plot(pts[:, 0], pts[:, 1], color=tl.NEUTRAL, lw=6, solid_capstyle="round",
               label="target letter")
        for name in order:
            P = arr.get(f"path|{cond.name}|{name}")
            if P is not None:
                a.plot(P[:, 0], P[:, 1], color=tl.COLOR[name], ls=tl.LS[name], lw=1.5,
                       label=tl.LABEL[name])
        _limits(a, sc, pts)
        a.set_title(cond.label(), fontsize=9)
        a.set_xlabel("x (m)")
    ax[0, 0].set_ylabel("y (m)")
    fig.tight_layout()
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8)
    tl.title(fig, "Experiment 3: path of the arm tip vs the target letter "
                  "(lines: controllers; panels: conditions)",
             f"{sc.what(letter)}; torque limit {sc.torque_limit} N·m; one variable "
             "changed per panel", y=1.1)
    fig.savefig(f"{FIG}/tracking_paths_{sc.tag(letter)}.png"); plt.close(fig)


def fig_timeseries(sc, letter, arr, order):
    """Main condition: tip error and both torques vs time."""
    fig, ax = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    for name in order:
        err = arr.get(f"err|main|{name}")
        U = arr.get(f"u|main|{name}")
        if err is None:
            continue
        tt = np.arange(len(err)) * cfg.DT
        kw = dict(color=tl.COLOR[name], ls=tl.LS[name], lw=1.5)
        ax[0].plot(tt, 1000 * err, label=tl.LABEL[name], **kw)
        ax[1].plot(tt, U[:, 0], **kw)
        ax[2].plot(tt, U[:, 1], **kw)
    for a, j in ((ax[1], 0), (ax[2], 1)):
        for sgn in (1, -1):
            a.axhline(sgn * sc.torque_limit[j], color=tl.TEXT2, ls=":", lw=0.8)
        a.set_ylabel(f"$u_{j + 1}$ (N·m)")
    for a in ax:
        a.axvline(sc.draw_time, color=tl.TEXT2, ls="--", lw=0.8)
    ax[0].annotate("letter finished, holding", (sc.draw_time, 1), xytext=(4, 4),
                   textcoords="offset points", fontsize=8, color=tl.TEXT2)
    ax[0].set_ylabel("tip error (mm)"); ax[0].set_yscale("log")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    tl.title(fig, f"Experiment 3: tip error and joint torques vs time ({sc.what(letter)})",
             describe(tl.MAIN, sc), y=1.05)
    fig.savefig(f"{FIG}/tracking_timeseries_{sc.tag(letter)}.png"); plt.close(fig)


def gif(sc, letter, arr, order, fps=25):
    """Main condition: every controller's arm drawing the letter, side by side."""
    names = [n for n in order if f"x|main|{n}" in arr]
    if not names:
        print("  no joint data saved (rerun without --figures); GIF skipped")
        return
    l1 = cfg.LENGTH[0]
    X = {n: np.nan_to_num(arr[f"x|main|{n}"]) for n in names}
    P = {n: tl.forward_kinematics(X[n]) for n in names}
    N = min(len(v) for v in X.values())
    idx = np.arange(0, N, max(1, int(round(1.0 / (fps * cfg.DT)))))
    ncol = min(4, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.9 * nrow + 0.5), squeeze=False)
    pts = arr["letter"]
    arms, trails = {}, {}
    for a, n in zip(axs.flat, names):
        _workspace(a)
        a.plot(pts[:, 0], pts[:, 1], color=tl.NEUTRAL, lw=5, solid_capstyle="round")
        trails[n], = a.plot([], [], color=tl.COLOR[n], lw=1.8)
        arms[n], = a.plot([], [], "o-", color=tl.TEXT, lw=2.5, ms=4)
        L = 1.1 * reach()[1]                            # whole arm always visible
        a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title(tl.LABEL[n], fontsize=8, color=tl.COLOR[n])
    for a in list(axs.flat)[len(names):]:
        a.axis("off")
    head = fig.suptitle("", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(i):
        j = idx[i]
        for n in names:
            q1 = X[n][j, 0]
            elbow = (l1 * np.sin(q1), -l1 * np.cos(q1))
            tip = P[n][j]
            arms[n].set_data([0, elbow[0], tip[0]], [0, elbow[1], tip[1]])
            trails[n].set_data(P[n][:j + 1, 0], P[n][:j + 1, 1])
        phase = "drawing" if j * cfg.DT <= sc.draw_time else "holding"
        head.set_text(f"Experiment 3: {sc.what(letter)} (main condition)\n"
                      f"t = {j * cfg.DT:4.2f} s, {phase}")
        return [*arms.values(), *trails.values(), head]

    anim = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
    path = f"{FIG}/tracking_{sc.tag(letter)}.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {path}")


def table(sc, letter, rows, order):
    md = [f"Experiment 3 ({sc.what(letter)}; torque limit {sc.torque_limit} N·m). "
          "Shape error in mm (mean distance from the drawn path to the letter); "
          "one variable changed per row.", "",
          "| condition | " + " | ".join(tl.SHORT[n] for n in order) + " |",
          "|" + "---|" * (len(order) + 1)]
    for cond in tracking_conditions():
        md.append(f"| {cond.label()} | " + " | ".join(_cell(rows, cond.name, n)
                                                      for n in order) + " |")
    md.append("| gain unstable along the path, main (% of points) | " + " | ".join(
        _unstable(rows, n) for n in order) + " |")
    md += ["", "Last row: frozen-time check of each controller's effective gain "
           "K_x + K_extra dpsi/dx at points along the path, on the true arm's "
           "Jacobians (the tracking version of the balance suitability check). "
           "Non-zero means the gain stops stabilizing the arm somewhere on the letter."]
    open(os.path.join(sc.out_dir(letter), "summary_tracking.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def table_all(scenes):
    """Every scene and letter side by side: the small-vs-large answer.
    Shape error (mm) per condition; lower is better."""
    data = []
    for sc in scenes:
        for letter in sc.letters:
            if os.path.exists(os.path.join(sc.out_dir(letter), "results.csv")):
                data.append((sc, letter) + _load(sc, letter))
    if not data:
        return
    order = data[0][4]
    md = ["Experiment 3: shape error in mm (mean distance to the letter; lower is better). "
          "Small scene = near upright; large scene = around the shoulder.", "",
          "| scene, letter | condition | " + " | ".join(tl.SHORT[n] for n in order)
          + " | best |",
          "|" + "---|" * (len(order) + 3)]
    for sc, letter, rows, _, _ in data:
        for cond in tracking_conditions():
            vals = {n: [q for q in rows if q["condition"] == cond.name
                        and q["controller"] == n] for n in order}
            ok = {n: v[0]["shape_mm"] for n, v in vals.items() if v and not v[0]["diverged"]}
            best = tl.SHORT[min(ok, key=ok.get)] if ok else "—"
            md.append(f"| {sc.name}, {letter} | {cond.label()} | "
                      + " | ".join(_cell(rows, cond.name, n) for n in order)
                      + f" | {best} |")
        md.append(f"| {sc.name}, {letter} | gain unstable along path (main) | "
                  + " | ".join(_unstable(rows, n) for n in order) + " | |")
    path = os.path.join("results", "tracking", tl.TAG, "summary_scenes.md")
    open(path, "w").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


def figures(sc, letter, make_gif=True):
    rows, arr, order = _load(sc, letter)
    fig_paths(sc, letter, arr, order)
    fig_timeseries(sc, letter, arr, order)
    table(sc, letter, rows, order)
    if make_gif:
        gif(sc, letter, arr, order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=list(cfg.TRACK_RUN),
                    choices=list(cfg.TRACK_SCENES),
                    help="scenes to run (default: TRACK_RUN in config.py)")
    ap.add_argument("--figures", action="store_true", help="only redraw from saved results")
    ap.add_argument("--check", action="store_true", help="only check the letters are feasible")
    ap.add_argument("--no-gif", action="store_true", help="skip the GIFs")
    a = ap.parse_args()
    scenes = [Scene(s) for s in a.scenes]
    if a.check:
        check(scenes)
        return
    for sc in scenes:
        for letter in sc.letters:
            if not a.figures:
                run_letter(sc, letter)
            figures(sc, letter, make_gif=not a.no_gif)
    table_all(scenes)


if __name__ == "__main__":
    main()