"""Vectorized closed-loop sweep (all initial states simulated at once).

Same plant (dynamics_np, checked against DFKI to ~1e-13), same RK4 step,
same record length and the same metrics as the DFKI sweep, but ~30x faster.
Ideal sensing only (no noise / delay).  Selected per condition through
config.MAIN_SIMULATOR / ROBUSTNESS_SIMULATOR.
"""
import numpy as np

import config as cfg
import experiments as ex
from dynamics_np import VecPlant
from metrics import evaluate
from plant_utils import wrap_angle
from sweep import KEYS

N_STEPS = 1001          # DFKI records 1001 control steps for t in [0, 5) s


def simulate_batch(K, lift, u_star, X0, n_steps=N_STEPS):
    vp = VecPlant()                            # reads cfg.TORQUE_LIMIT now
    X = np.zeros((len(X0), n_steps + 1, 4)); U = np.zeros((len(X0), n_steps, 2))
    x = np.array(X0, dtype=float); X[:, 0] = x
    with np.errstate(all="ignore"):
        for k in range(n_steps):
            dx = x - cfg.X_STAR
            dx[:, :2] = wrap_angle(dx[:, :2])
            u = vp.sat(u_star - lift(dx) @ K.T)
            u = np.where(np.isfinite(u), u, 0.0)
            x = vp.rk4_step(x, u)
            x = np.where(np.abs(x) < 1e6, x, np.nan)     # diverged runs -> nan
            X[:, k + 1] = x; U[:, k] = u
    T = np.arange(n_steps + 1) * cfg.DT
    return T, X, U


def run_points(K, lift, u_star, X0):
    """Metrics for a list of initial states (under the current cfg)."""
    T, X, U = simulate_batch(K, lift, u_star, X0)
    out = {key: np.zeros(len(X0)) for key in KEYS}
    for i in range(len(X0)):
        Xi = np.nan_to_num(X[i], nan=1e3)
        m = evaluate(T, Xi, U[i], u_star)
        for key in KEYS:
            out[key][i] = m[key]
    out["success"] = out["success"].astype(bool)
    return out


def run_grid(K, lift, u_star, cond):
    """Map over the initial-angle grid of `cond` (call inside ex.applied)."""
    g, X0 = ex.initial_states(cond)
    n = len(g)
    flat = run_points(K, lift, u_star, X0)
    out = {k: v.reshape(n, n) for k, v in flat.items()}
    out["grid"] = g
    return out


def identity_lift(dx):
    return dx
