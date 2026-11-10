"""Performance metrics computed on the simulated NONLINEAR trajectory."""
import numpy as np

import config as cfg
from plant_utils import wrap_angle


def convergence_rate(T, dX, Q=cfg.Q, e_hi=cfg.RATE_E_HI, e_lo=cfg.RATE_E_LO):
    """Exponential tail rate and transient constant of one trajectory.

    e(t) = sqrt(dx' Q dx).  alpha_hat = -slope of ln e(t), fitted from the
    first time e < e_hi until the first time e < e_lo (noise / numerics
    floor).  c_hat = max_t e(t) exp(alpha_hat t) / e(0) over that span, so
    e(t) <= c_hat e(0) exp(-alpha_hat t): the overshoot / saturation cost
    of reaching the exponential regime."""
    e = np.sqrt(np.einsum("ki,ij,kj->k", dX, Q, dX))
    t = np.asarray(T) - T[0]
    below_hi = np.where(e < e_hi)[0]
    if len(below_hi) == 0:
        return np.nan, np.nan
    k1 = below_hi[0]
    below_lo = np.where(e[k1:] < e_lo)[0]
    k2 = k1 + (below_lo[0] if len(below_lo) else len(e) - k1)
    if k2 - k1 < 20 or e[0] <= 0:
        return np.nan, np.nan
    slope = np.polyfit(t[k1:k2], np.log(e[k1:k2]), 1)[0]
    alpha = -slope
    c = float(np.max(e[:k2] * np.exp(alpha * t[:k2])) / e[0])
    return float(alpha), c


def evaluate(T, X, U, u_star, x_star=cfg.X_STAR, Q=cfg.Q, R=cfg.R,
             torque_limit=None, angle_tol=cfg.ANGLE_TOL,
             vel_tol=cfg.VEL_TOL, hold_time=cfg.HOLD_TIME):
    T = np.asarray(T)
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    n = min(len(X), len(U))          # pair x_k with the u_k applied at x_k
    T, X, U = T[:n], X[:n], U[:n]
    dt = T[1] - T[0]

    dX = X - x_star
    dX[:, :2] = wrap_angle(dX[:, :2])
    dU = U - u_star

    finite = np.all(np.isfinite(dX))
    # "At rest" is judged on velocity averaged over VEL_AVG_WINDOW: with
    # realistic sensing, encoder quantization fed through K makes the true
    # velocity jitter around zero even when the arm is held still.
    w = max(1, int(round(cfg.VEL_AVG_WINDOW / dt)))
    kern = np.ones(w) / w
    vel_avg = np.column_stack([np.convolve(dX[:, j], kern, mode="same") for j in (2, 3)])
    inside = (np.all(np.abs(dX[:, :2]) < angle_tol, axis=1)
              & np.all(np.abs(vel_avg) < vel_tol, axis=1))

    # settling time = first time after which the state never leaves the box
    if finite and inside[-1]:
        outside = np.where(~inside)[0]
        k_settle = 0 if len(outside) == 0 else outside[-1] + 1
        t_settle = T[k_settle] - T[0]
        success = (T[-1] - T[k_settle]) >= hold_time
    else:
        t_settle, success = np.nan, False

    alpha, c = convergence_rate(T, dX) if success else (np.nan, np.nan)

    if torque_limit is None:          # read at call time: conditions change it
        torque_limit = cfg.TORQUE_LIMIT
    stage = (np.einsum("ki,ij,kj->k", dX, Q, dX)
             + np.einsum("ki,ij,kj->k", dU, R, dU))
    lim = np.asarray(torque_limit)

    return {
        "success": bool(success),
        "settling_time": float(t_settle),
        # RMSE of the wrapped joint-angle errors over the whole run
        "rmse_q1": float(np.sqrt(np.mean(dX[:, 0] ** 2))),
        "rmse_q2": float(np.sqrt(np.mean(dX[:, 1] ** 2))),
        "rmse_q": float(np.sqrt(np.mean(np.sum(dX[:, :2] ** 2, axis=1)))),
        # common physical quadratic cost, integrated in time
        "cost_J": float(np.sum(stage) * dt),
        # control effort  int ||u||^2 dt  (absolute torque, not deviation)
        "effort": float(np.sum(U ** 2) * dt),
        # fraction of steps with at least one joint at its torque limit
        "sat_frac": float(np.mean(np.any(np.abs(U) >= lim - 1e-9, axis=1))),
        "max_angle_err": float(np.max(np.abs(dX[:, :2]))),
        # exponential tail rate (1/s) and transient constant (successes only)
        "alpha_hat": alpha,
        "c_hat": c,
    }
