"""Stage 1-2: verify the plant and build the local-linearization LQR baseline.

    python run_local_lqr.py              # design, 5 test runs, figures, GIF
    python run_local_lqr.py --map        # + initial-angle map (Experiment 1)
    python run_local_lqr.py --live       # + DFKI live animation of one run
"""
import argparse
import os

import numpy as np
import matplotlib

import config as cfg
from plant_utils import (make_plant, make_simulator, equilibrium_input,
                         linearize, c2d_zoh)
from lqr_design import dlqr
from controllers import LiftedLQRController
from metrics import evaluate

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
np.set_printoptions(precision=4, suppress=True)
TAG = f"{cfg.PLANT_PROFILE}-{cfg.REALISM}"


def design_local_lqr():
    """Design on the friction-free design model (see make_plant)."""
    dplant = make_plant(for_design=True)
    u_star = equilibrium_input(dplant, cfg.X_STAR)
    Ac, Bc = linearize(dplant, cfg.X_STAR, u_star, check=True)
    Ad, Bd = c2d_zoh(Ac, Bc, cfg.DT)
    K, P, eig_cl = dlqr(Ad, Bd, cfg.Q, cfg.R, check=True)
    return dict(u_star=u_star, Ac=Ac, Bc=Bc, Ad=Ad, Bd=Bd, K=K, P=P, eig_cl=eig_cl)


def local_lqr_factory():
    """Picklable factory used by sweep.run_map."""
    d = design_local_lqr()
    return (LiftedLQRController(d["K"], cfg.X_STAR, d["u_star"],
                                cfg.TORQUE_LIMIT, name="local LQR"), d["u_star"])


def simulate(plant, controller, x0, animate=False, seed=cfg.SEED):
    sim = make_simulator(plant, seed=seed)
    controller.init()
    run = sim.simulate_and_animate if animate else sim.simulate
    T, X, U = run(t0=0.0, x0=list(x0), tf=cfg.T_FINAL, dt=cfg.DT,
                  controller=controller, integrator=cfg.INTEGRATOR)
    return np.asarray(T), np.asarray(X), np.asarray(U)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", action="store_true", help="run the initial-angle sweep")
    ap.add_argument("--live", action="store_true", help="DFKI live animation")
    ap.add_argument("--n", type=int, default=cfg.MAP_N, help="map grid size")
    args = ap.parse_args()
    if not args.live:
        matplotlib.use("Agg")
    import viz   # after the backend is chosen

    plant = make_plant()                     # simulation (truth) plant
    d = design_local_lqr()

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
    viz.draw_conventions("figures/conventions.png")

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
        T, X, U = simulate(plant, ctrl, x0)
        m = evaluate(T, X, U, d["u_star"])
        print(f"{name:8s} {str([d1, d2]):>16s} {('yes' if m['success'] else 'NO'):>4s} "
              f"{m['settling_time']:7.2f} {m['rmse_q']:7.3f} {m['cost_J']:8.2f} "
              f"{m['effort']:8.2f} {100 * m['sat_frac']:6.1f}")
        viz.plot_run(T, X, U, f"Local LQR, {name}: dq0 = {[d1, d2]}  [{TAG}]",
                     f"figures/local_lqr_{TAG}_{name}.png")
        runs.append((f"{name} {[d1, d2]}", T, X, U))
        if name == "large":
            viz.save_animation(T, X, U, f"figures/local_lqr_{TAG}_{name}.gif",
                               title=f"Local LQR, dq0 = {[d1, d2]}")
    viz.plot_overlay(runs, f"Local LQR, growing initial offsets  [{TAG}]",
                     f"figures/local_lqr_{TAG}_overlay.png")

    if args.map:
        from sweep import run_map
        print(f"\n=== Initial-angle map {args.n}x{args.n} ===")
        res = run_map(local_lqr_factory, n=args.n)
        np.savez(f"results/map_local_lqr_{TAG}.npz", **res)
        print(f"success on slice: {100 * res['success'].mean():.1f}%")
        viz.plot_map(res, f"Local-linearization LQR  [{TAG}]",
                     f"figures/map_local_lqr_{TAG}.png")

    if args.live:
        simulate(plant, ctrl, cfg.X_STAR + np.array([0.5, -0.5, 0, 0]), animate=True)


if __name__ == "__main__":
    main()