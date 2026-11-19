"""SIMULATION 1 of 2 (balance): local-linearization LQR.

Design: linearize the plant at x*, discretize (zero-order hold), solve the
discrete Riccati equation -> u = sat(u* - K dx).  Then simulate it over the
experiment conditions defined in config.py, section 1 (the same conditions
run_koopman.py uses for the Koopman controllers).

    python run_local_lqr.py               # design + 5 test runs + figures + GIF
    python run_local_lqr.py --maps main   # + Experiment 1 map (initial angles)
    python run_local_lqr.py --maps robust # + robustness maps (config.ROBUSTNESS)
    python run_local_lqr.py --maps all    # + both
    python run_local_lqr.py --live        # + DFKI live animation of one run
"""
import argparse
import os

import numpy as np
import matplotlib

import config as cfg
import tool_lib as tl
from plant import make_plant
from control import design_local_lqr, local_lqr_factory, LiftedLQRController

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
np.set_printoptions(precision=4, suppress=True)
TAG = tl.TAG


def run_maps(which):
    """Simulate local LQR from every initial state of every condition."""
    print(f"\n=== Local LQR maps: {which} conditions, {cfg.MAP_N} x {cfg.MAP_N} "
          f"initial angles each ===")
    for cond in tl.conditions(which):
        res = tl.run_map(local_lqr_factory, cond)
        path = tl.save_map("local", cond, res)
        print(f"{cond.name:16s} [{res['simulator']}]  success "
              f"{100 * res['success'].mean():5.1f}%   ({cond.describe()})  -> {path}",
              flush=True)
        what = ("Experiment 1: initial angles varied. Where does local LQR stabilize the arm?"
                if cond.group == "main" else
                f"Robustness ({cond.label()}): initial angles varied, local LQR")
        tl.plot_map(res, f"{what}\nFixed: {cond.describe()}",
                     f"figures/map_local__{cond.name}_{TAG}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", choices=["main", "robust", "all"],
                    help="simulate the experiment conditions (config.py, section 1)")
    ap.add_argument("--live", action="store_true", help="DFKI live animation")
    args = ap.parse_args()
    if not args.live:
        matplotlib.use("Agg")

    plant = make_plant()                     # simulation (truth) plant
    d = design_local_lqr()
    main_cond = tl.MAIN.describe()

    print(f"=== Plant / target  [{TAG}] ===")
    print("mass", cfg.MASS, " length", cfg.LENGTH, " com", np.round(cfg.COM, 4))
    print("torque limit", np.round(cfg.TORQUE_LIMIT, 3), " dt", cfg.DT)
    print("x* =", cfg.X_STAR, "  u* =", d["u_star"])
    print("open-loop eig(A_c) =", np.linalg.eigvals(d["Ac"]))
    print("closed-loop max|eig(A_d - B_d K)| =", np.max(np.abs(d["eig_cl"])))
    print("K =\n", d["K"])
    np.savez(f"results/local_lqr_{TAG}.npz", x_star=cfg.X_STAR, dt=cfg.DT,
             Q=cfg.Q, R=cfg.R, **d)

    ctrl = LiftedLQRController(d["K"], cfg.X_STAR, d["u_star"],
                               cfg.TORQUE_LIMIT, name="local LQR")
    tl.draw_conventions("figures/conventions.png")

    # Test near the target first, then move outwards (velocities = 0).
    # the values are [dq1, dq2] offsets from the target angles in radians.
    offsets = [("tiny", [0.05, -0.05]), ("small", [0.20, -0.20]),
               ("medium", [0.50, -0.50]), ("large", [1.00, -1.00]),
               ("huge", [2.00, 1.50])]
    print("\n=== Closed-loop tests on the nonlinear DFKI plant ===")
    print(f"{'case':8s} {'dq0':>16s} {'ok':>4s} {'t_s[s]':>7s} {'RMSE_q':>7s} "
          f"{'J':>8s} {'effort':>8s} {'sat%':>6s}")
    runs = []
    for name, (d1, d2) in offsets:
        x0 = cfg.X_STAR + np.array([d1, d2, 0.0, 0.0])
        T, X, U = tl.simulate(ctrl, x0, plant=plant)
        m = tl.evaluate(T, X, U, d["u_star"])
        print(f"{name:8s} {str([d1, d2]):>16s} {('yes' if m['success'] else 'NO'):>4s} "
              f"{m['settling_time']:7.2f} {m['rmse_q']:7.3f} {m['cost_J']:8.2f} "
              f"{m['effort']:8.2f} {100 * m['sat_frac']:6.1f}")
        tl.plot_run(T, X, U,
                     f"Local LQR test run from initial angles dq0 = {[d1, d2]} rad"
                     f"\nFixed: {main_cond}",
                     f"figures/local_lqr_{TAG}_{name}.png")
        runs.append((f"dq0 = {[d1, d2]} rad", T, X, U))
        if name == "large":
            tl.save_animation(T, X, U, f"figures/local_lqr_{TAG}_{name}.gif",
                               title=f"Local LQR, dq0 = {[d1, d2]} rad")
    tl.plot_overlay(runs, "Local LQR: initial angle offset grows, convergence slows"
                     f"\nFixed: {main_cond}",
                     f"figures/local_lqr_{TAG}_overlay.png")

    if args.maps:
        run_maps(args.maps)

    if args.live:
        tl.simulate(ctrl, cfg.X_STAR + np.array([0.5, -0.5, 0, 0]), plant=plant, animate=True)


if __name__ == "__main__":
    main()
