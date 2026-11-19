"""Tools shared by every experiment: what is varied, how runs are scored,
how maps are swept, the tracking references, and all plotting.

  Conditions   Condition, MAIN, conditions(), applied(), initial_states()
               (config.py section 1 -> named conditions shared by every script)
  Simulation   simulate() (one DFKI run), run_map() (initial-angle map, DFKI or fast)
  Evaluation   evaluate() (balance metrics), tracking_metrics() (tip errors)
  Tracking     letter paths, minimum-jerk timing, kinematics, build_reference()
  Plotting     COLOR / LS / MARKER / LABEL per controller, apply_style(), title(),
               plot_run(), plot_overlay(), plot_map(), save_animation(), ...
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap, LogNorm
from scipy.spatial import cKDTree

import config as cfg
from plant import make_plant, make_simulator, state_error, wrap_angle, VecPlant


# ======================================================================
# Experimental conditions (config.py section 1)
# ======================================================================
BASE_TORQUE = tuple(cfg.TORQUE_LIMIT)          # main torque limit of this profile
TAG = f"{cfg.PLANT_PROFILE}-{cfg.REALISM}"
SENSING = "ideal actuation" if cfg.REALISM == "ideal" else "realistic actuation"
GROUPS = ("torque", "velocity", "model", "plant", "noise")


def _fmt(values):
    return ", ".join(f"{v:g}" for v in values)


@dataclass(frozen=True)
class Condition:
    group: str                     # "main" | "torque" | "velocity" | "model" | "plant" | "noise"
    torque_scale: float = 1.0
    v0: tuple = tuple(cfg.INIT_VELOCITY_MAIN)
    model_error: float = cfg.MODEL_ERROR_MAIN
    plant_mass: float = cfg.PLANT_MASS_MAIN
    noise_level: float = cfg.NOISE_LEVEL_MAIN

    @property
    def torque_limit(self):
        return [round(self.torque_scale * t, 4) for t in BASE_TORQUE]

    @property
    def name(self):
        return {"main": "main",
                "torque": f"torque_x{self.torque_scale:.2f}",
                "velocity": f"vel_{self.v0[0]:+.1f}_{self.v0[1]:+.1f}",
                "model": f"model{_model_tag()}_x{self.model_error:.2f}",
                "plant": f"plant_x{self.plant_mass:.2f}",
                "noise": f"noise_x{self.noise_level:.2f}"}[self.group]

    @property
    def simulator(self):
        sim = cfg.MAIN_SIMULATOR if self.group == "main" else cfg.ROBUSTNESS_SIMULATOR
        if sim == "fast" and (cfg.REALISM != "ideal" or self.noise_level > 0):
            return "dfki"                       # the fast copy has no noise / delay
        return sim

    def describe(self):
        """All fixed values of this condition, for figure titles."""
        parts = [f"torque limit [{_fmt(self.torque_limit)}] N·m",
                 f"initial velocity ({_fmt(self.v0)}) rad/s",
                 "exact design model" if self.model_error == 1.0 else
                 f"{_model_what()} ×{self.model_error:g}",
                 "nominal arm" if self.plant_mass == 1.0 else
                 f"arm masses ×{self.plant_mass:g}",
                 "no sensor noise" if self.noise_level == 0 else
                 f"sensor noise ×{self.noise_level:g}",
                 SENSING]
        return ", ".join(parts)

    def label(self):
        """Short name of what this condition changes, for legends / tables."""
        return {"main": "main",
                "torque": f"torque ×{self.torque_scale:g} [{_fmt(self.torque_limit)}] N·m",
                "velocity": f"velocity ({_fmt(self.v0)}) rad/s",
                "model": f"{_model_what()} ×{self.model_error:g}",
                "plant": f"arm masses ×{self.plant_mass:g}",
                "noise": f"sensor noise ×{self.noise_level:g}"}[self.group]

    @property
    def value(self):
        """The one value this condition changes (for plotting)."""
        return {"main": None, "torque": self.torque_scale, "velocity": self.v0,
                "model": self.model_error, "plant": self.plant_mass,
                "noise": self.noise_level}[self.group]


MAIN = Condition("main")


def _model_what():
    """What the model error scales, for labels (config.MODEL_ERROR_LINKS)."""
    return {"both": "design-model masses", "shoulder": "design-model shoulder mass",
            "elbow": "design-model elbow mass"}[cfg.MODEL_ERROR_LINKS]


def _model_tag():
    return "" if cfg.MODEL_ERROR_LINKS == "both" else f"_{cfg.MODEL_ERROR_LINKS}"


def torque_conditions():
    return [Condition("torque", s) for s in cfg.TORQUE_SCALE_SWEEP]


def velocity_conditions():
    return [Condition("velocity", v0=tuple(float(x) for x in v)) for v in cfg.INIT_VELOCITY_SWEEP]


def model_conditions():
    return [Condition("model", model_error=float(e)) for e in cfg.MODEL_ERROR_SWEEP]


def plant_conditions():
    return [Condition("plant", plant_mass=float(m)) for m in cfg.PLANT_MASS_SWEEP]


def noise_conditions():
    return [Condition("noise", noise_level=float(n)) for n in cfg.NOISE_LEVEL_SWEEP]


def conditions(which="all", groups=None):
    """which: "main" | "robust" | "all"; groups: robustness groups to include
    (default: config.ROBUSTNESS)."""
    groups = cfg.ROBUSTNESS if groups is None else groups
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        raise ValueError(f"unknown robustness group(s) {bad}; choose from {GROUPS}")
    makers = {"torque": torque_conditions, "velocity": velocity_conditions,
              "model": model_conditions, "plant": plant_conditions,
              "noise": noise_conditions}
    robust = [c for g in groups for c in makers[g]()]
    return {"main": [MAIN], "robust": robust, "all": [MAIN] + robust}[which]


def apply_condition(cond):
    """Set this condition's variables in config.  Everything reads them when
    it is built: plant and simulator (TORQUE_LIMIT, PLANT_MASS, MEAS_NOISE),
    physics-based designs (MODEL_ERROR), controllers and metrics (TORQUE_LIMIT)."""
    cfg.TORQUE_LIMIT = list(cond.torque_limit)
    cfg.MODEL_ERROR = float(cond.model_error)
    cfg.PLANT_MASS = float(cond.plant_mass)
    cfg.MEAS_NOISE = list(cond.noise_level * np.asarray(cfg.NOISE_BASE))


@contextmanager
def applied(cond):
    old = (list(cfg.TORQUE_LIMIT), cfg.MODEL_ERROR, cfg.PLANT_MASS, list(cfg.MEAS_NOISE))
    apply_condition(cond)
    try:
        yield cond
    finally:
        cfg.TORQUE_LIMIT, cfg.MODEL_ERROR, cfg.PLANT_MASS, cfg.MEAS_NOISE = old


def initial_states(cond, n=cfg.MAP_N, rng=cfg.MAP_RANGE):
    """Grid of initial angles (Experiment 1) at this condition's velocity.
    Row i <-> dq2 = g[i], column j <-> dq1 = g[j]."""
    g = np.linspace(-rng, rng, n)
    D1, D2 = np.meshgrid(g, g)
    X0 = cfg.X_STAR + np.column_stack([D1.ravel(), D2.ravel(),
                                       np.full(D1.size, cond.v0[0]),
                                       np.full(D1.size, cond.v0[1])])
    return g, X0


def map_path(controller, cond):
    return os.path.join("results", "maps", TAG, f"{controller}__{cond.name}.npz")


def save_map(controller, cond, res):
    path = map_path(controller, cond)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **res, controller=controller, condition=cond.name,
             torque_limit=np.array(cond.torque_limit), v0=np.array(cond.v0),
             model_error=cond.model_error, plant_mass=cond.plant_mass,
             noise_level=cond.noise_level)
    return path


def load_map(controller, cond):
    path = map_path(controller, cond)
    if not os.path.exists(path):
        hint = ("python run_local_lqr.py --maps all" if controller == "local"
                else "python run_koopman.py --stage maps")
        raise FileNotFoundError(f"{path} is missing; run: {hint}")
    return dict(np.load(path))


# ======================================================================
# Balance metrics
# ======================================================================
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


# ======================================================================
# Simulation: one run, and initial-angle maps
# ======================================================================
def identity_lift(dx):
    """psi(dx) = dx: local LQR's 'lift'."""
    return dx


def simulate(controller, x0, tf=None, plant=None, seed=None, animate=False):
    """One run in DFKI's simulator on the true plant of the current condition."""
    plant = make_plant() if plant is None else plant
    sim = make_simulator(plant, seed=cfg.SEED if seed is None else seed)
    controller.init()
    run = sim.simulate_and_animate if animate else sim.simulate
    T, X, U = run(t0=0.0, x0=list(x0), tf=cfg.T_FINAL if tf is None else float(tf),
                  dt=cfg.DT, controller=controller, integrator=cfg.INTEGRATOR)
    return np.asarray(T), np.asarray(X), np.asarray(U)


KEYS = ["success", "settling_time", "rmse_q", "cost_J", "effort", "sat_frac",
        "alpha_hat", "c_hat"]
_W = {}


def _init(factory, cond):
    # Applied inside the worker: on macOS / Windows workers start fresh and
    # would otherwise silently use the main torque limit.
    apply_condition(cond)
    _W["plant"] = make_plant()
    _W["ctrl"], _W["u_star"] = factory()


def _run_cell(args):
    k, x0 = args
    sim = make_simulator(_W["plant"], seed=cfg.SEED + k)
    _W["ctrl"].init()
    T, X, U = sim.simulate(t0=0.0, x0=list(x0), tf=cfg.T_FINAL, dt=cfg.DT,
                           controller=_W["ctrl"], integrator=cfg.INTEGRATOR)
    m = evaluate(T, X, U, _W["u_star"])
    return k, [m[key] for key in KEYS]


def _run_map_dfki(factory, cond, workers=None):
    g, X0 = initial_states(cond)
    n = len(g)
    out = {key: np.zeros((n, n)) for key in KEYS}
    with Pool(workers or os.cpu_count(), initializer=_init,
              initargs=(factory, cond)) as p:
        for k, vals in p.imap_unordered(_run_cell, list(enumerate(X0)), chunksize=4):
            for key, v in zip(KEYS, vals):
                out[key][k // n, k % n] = v
    out["success"] = out["success"].astype(bool)
    out["grid"] = g
    return out


def run_map(factory, cond, simulator=None):
    """Map of one controller under one condition, with the condition's
    simulator unless `simulator` is given ("dfki" | "fast")."""
    sim = simulator or cond.simulator
    if sim == "fast" and cfg.REALISM != "ideal":
        sim = "dfki"
    if sim == "fast":
        with applied(cond):
            ctrl, u_star = factory()
            res = run_grid(ctrl.K, ctrl.lift, u_star, cond)
    else:
        res = _run_map_dfki(factory, cond)
    res["simulator"] = sim
    return res


N_STEPS = 1001          # DFKI records 1001 control steps for t in [0, 5) s


def simulate_batch(K, lift, u_star, X0, n_steps=N_STEPS):
    vp = VecPlant(mass_scale=cfg.PLANT_MASS)   # the TRUE plant of this condition
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
    """Map over the initial-angle grid of `cond` (call inside applied())."""
    g, X0 = initial_states(cond)
    n = len(g)
    flat = run_points(K, lift, u_star, X0)
    out = {k: v.reshape(n, n) for k, v in flat.items()}
    out["grid"] = g
    return out


# ======================================================================
# Tracking references (Experiment 3)
# ======================================================================
ELBOW = 1.0          # +1 or -1: which elbow branch the inverse kinematics uses


# ------------------------------------------------------------ letter shapes
def letter_segments(letter):
    """Letter of unit height centred at (0, 0): list of (polyline, pen_down).
    Pen-up segments are moves between strokes; they are drawn (the arm has
    to move) but excluded from the shape error."""
    if letter == "C":                        # arc open to the right, 260 degrees
        th = np.radians(np.linspace(50, 310, 400))
        return [(0.5 * np.column_stack([np.cos(th), np.sin(th)]), True)]
    if letter == "O":                        # full circle, starting at the top
        th = np.radians(np.linspace(90, 450, 500))
        return [(0.5 * np.column_stack([np.cos(th), np.sin(th)]), True)]
    if letter == "G":                        # arc from top right round to the right,
        th = np.radians(np.linspace(45, 360, 450))     # then a bar inwards (a corner);
        return [(0.5 * np.column_stack([np.cos(th), np.sin(th)]), True),  # the bar stops
                (np.array([[0.5, 0.0], [0.25, 0.0]]), True)]               # at half radius
    if letter == "X":                        # two strokes with sharp corners
        w = 0.35
        return [(np.array([[-w, 0.5], [w, -0.5]]), True),
                (np.array([[w, -0.5], [w, 0.5]]), False),
                (np.array([[w, 0.5], [-w, -0.5]]), True)]
    raise ValueError(f"unknown letter {letter!r}; use C, G, O or X")


def letter_size(size):
    """Letter height (m): a number, or "auto" = the letter's ring runs through
    the middle of the reachable ring |l1-l2| .. l1+l2 (radius max(l1, l2))."""
    if size == "auto":
        return 2.0 * max(cfg.LENGTH)
    return float(size)


def reach():
    """Tip distance from the shoulder the arm can reach (m): (min, max)."""
    l1, l2 = cfg.LENGTH
    return abs(l1 - l2), l1 + l2


def _place(pts, size, center):
    return np.asarray(center, float) + size * pts


def letter_points(letter, size, center, n=2000):
    """Dense pen-down points of the target letter (for the shape error)."""
    out = []
    for poly, pen in letter_segments(letter):
        if pen:
            out.append(_resample(_place(poly, size, center), np.linspace(0, 1, n)))
    return np.concatenate(out)


def _resample(poly, frac):
    """Points at the given fractions of the polyline's arc length."""
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.clip(frac, 0, 1) * s[-1]
    return np.column_stack([np.interp(target, s, poly[:, 0]),
                            np.interp(target, s, poly[:, 1])])


def _min_jerk(tau):
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


# ------------------------------------------------------------ kinematics
def forward_kinematics(q):
    """q (N,2+) -> tip position (N,2)."""
    q1, q2 = q[:, 0], q[:, 1]
    l1, l2 = cfg.LENGTH
    return np.column_stack([l1 * np.sin(q1) + l2 * np.sin(q1 + q2),
                            -l1 * np.cos(q1) - l2 * np.cos(q1 + q2)])


def inverse_kinematics(p):
    """Tip positions (N,2) -> joint angles (N,2), elbow branch ELBOW.
    Raises if a point is out of reach."""
    l1, l2 = cfg.LENGTH
    x, y = p[:, 0], p[:, 1]
    c2 = (x ** 2 + y ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    if np.any(np.abs(c2) > 1.0):
        r = np.hypot(x, y)
        raise ValueError(f"letter out of reach: tip distance {r.min():.3f}-{r.max():.3f} m, "
                         f"arm reaches {abs(l1 - l2):.2f}-{l1 + l2:.2f} m; "
                         "change the scene's centre or size (config.TRACK_SCENES)")
    q2 = ELBOW * np.arccos(c2)
    # standard planar IK in angles a1 = q1 - pi/2 measured from +x
    a1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    q1 = np.unwrap(a1 + np.pi / 2)
    return np.column_stack([q1, q2])


# ------------------------------------------------------------ reference
def build_reference(letter, size, draw_time, center, hold=cfg.TRACK_HOLD, dt=cfg.DT):
    """Time-parametrized reference.  Returns a dict with
    T (N,), X (N,4) = [q, qd], QDD (N,2), P (N,2) tip path,
    pen (N,) pen down, draw (N,) still drawing (not the final hold),
    U_true (N,2) torque the path needs on the true plant, feasible (bool)."""
    segs = letter_segments(letter)
    lengths = np.array([np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) for p, _ in segs])
    P, pen = [], []
    for (poly, down), L in zip(segs, lengths):
        n = max(2, int(round(draw_time * L / lengths.sum() / dt)))
        frac = _min_jerk(np.linspace(0.0, 1.0, n + 1))      # rest at both ends
        pts = _resample(_place(poly, size, center), frac)
        if P:
            pts = pts[1:]                                  # joint shared with previous segment
        P.append(pts)
        pen.append(np.full(len(pts), down))
    P, pen = np.concatenate(P), np.concatenate(pen)
    n_draw = len(P)
    n_hold = int(round(hold / dt))
    P = np.concatenate([P, np.repeat(P[-1:], n_hold, axis=0)])
    pen = np.concatenate([pen, np.zeros(n_hold, bool)])
    draw = np.arange(len(P)) < n_draw

    q = inverse_kinematics(P)
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    qd[0] = qd[-1] = 0.0                                  # exactly at rest
    X = np.column_stack([q, qd])
    T = np.arange(len(P)) * dt

    true = VecPlant(coulomb_fric=[0.0, 0.0], mass_scale=cfg.PLANT_MASS)   # the true arm
    U_true = true.inverse_dynamics(X, qdd)
    lim = np.asarray(cfg.TORQUE_LIMIT)
    need = np.max(np.abs(U_true), axis=0)
    return dict(T=T, X=X, QDD=qdd, P=P, pen=pen, draw=draw, U_true=U_true,
                torque_need=need, feasible=bool(np.all(need <= lim)),
                letter=letter, size=size, draw_time=draw_time, center=tuple(center))


def max_angle_error(ref):
    """Largest |dq| from upright along the path (sets the Koopman fit region)."""
    dq = ref["X"][:, :2] - cfg.X_STAR[:2]
    dq = np.pi - np.mod(np.pi - dq, 2 * np.pi)
    return float(np.max(np.abs(dq)))


def tracking_metrics(T, X, U, ref, letter_pts):
    """Errors of the tip path against the reference (mm)."""
    n = min(len(X), len(U), len(ref["T"]))
    X, U = np.asarray(X)[:n], np.asarray(U)[:n]
    ok = np.all(np.isfinite(X))
    P = forward_kinematics(np.nan_to_num(X, nan=0.0))
    err = np.linalg.norm(P - ref["P"][:n], axis=1)
    on = ref["draw"][:n] & ref["pen"][:n]
    d, _ = cKDTree(letter_pts).query(P[on])
    lim = np.asarray(cfg.TORQUE_LIMIT)
    return dict(
        rmse_mm=1000 * float(np.sqrt(np.mean(err[ref["draw"][:n]] ** 2))),
        max_mm=1000 * float(np.max(err[ref["draw"][:n]])),
        shape_mm=1000 * float(np.mean(d)),        # distance to the letter, timing ignored
        final_mm=1000 * float(err[-1]),
        sat_pct=100 * float(np.mean(np.any(np.abs(U) >= lim - 1e-9, axis=1))),
        peak_u1=float(np.max(np.abs(U[:, 0]))), peak_u2=float(np.max(np.abs(U[:, 1]))),
        diverged=bool((not ok) or np.max(err) > 0.15),
    ), P


# ======================================================================
# Figure style
# ======================================================================
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GOOD, CRITICAL, NEUTRAL = "#0ca30c", "#d03b3b", "#bdbcb6"

ORDER = ["local", "poly1", "poly2", "poly3", "trig1", "trig2"]
COLOR = {"local": "#2a78d6", "poly1": "#eda100", "poly2": "#1baf7a",
         "poly3": "#eb6834", "trig1": "#e87ba4", "trig2": "#6250d6",
         "poly4": "#8a8984", "taylor": "#2a78d6",
         "local_fixed": "#2a78d6", "tvlqr": "#008300"}
LS = {"local": "-", "poly1": "--", "poly2": "-.", "poly3": ":",
      "trig1": "-", "trig2": "--", "poly4": ":", "taylor": "-",
      "local_fixed": "-", "tvlqr": "-."}
MARKER = {"local": "o", "poly1": "s", "poly2": "^", "poly3": "v",
          "trig1": "D", "trig2": "P", "poly4": "x", "taylor": "o",
          "local_fixed": "o", "tvlqr": "h"}
# Every label says which controller and, for Koopman, which observables (N)
LABEL = {"local": "Local LQR (Taylor model)",
         "taylor": "Taylor linearization (N=4)",
         "poly1": "Koopman, state only (N=4)",
         "poly2": "Koopman, quadratic (N=14)",
         "poly3": "Koopman, cubic (N=34)",
         "poly4": "Koopman, quartic (N=69)",
         "trig1": "Koopman, trig (N=8)",
         "trig2": "Koopman, trig (N=13)",
         "local_fixed": "Local LQR, fixed gain (upright model)",
         "tvlqr": "Local LQR, time-varying (TVLQR)"}
SHORT = {"local": "Local LQR", "poly1": "K: state (4)", "poly2": "K: quadratic (14)",
         "poly3": "K: cubic (34)", "trig1": "K: trig (8)", "trig2": "K: trig (13)",
         "local_fixed": "Local LQR fixed", "tvlqr": "TVLQR"}


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "lines.linewidth": 2.0, "lines.markersize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def title(fig, what, fixed, y=1.0):
    """Two-line figure title: line 1 = what is compared / varied,
    line 2 = what is held fixed."""
    fig.suptitle(f"{what}\nFixed: {fixed}", fontsize=10, y=y)


# ======================================================================
# Plotting
# ======================================================================
def _trim(T, X, U):
    n = min(len(T), len(X), len(U))
    return np.asarray(T)[:n], np.asarray(X)[:n], np.asarray(U)[:n]


# ------------------------------------------------ 1. time series
def plot_run(T, X, U, title, path, x_star=cfg.X_STAR):
    """Angle errors, velocities and torques vs time (figure type 1)."""
    T, X, U = _trim(T, X, U)
    dX = np.array([state_error(x, x_star) for x in X])
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(T, dX[:, 0], label=r"$q_1-q_1^\star$")
    ax[0].plot(T, dX[:, 1], label=r"$q_2-q_2^\star$")
    ax[0].axhspan(-cfg.ANGLE_TOL, cfg.ANGLE_TOL, color="g", alpha=0.12, lw=0)
    ax[0].set_ylabel("angle error [rad]")
    ax[1].plot(T, X[:, 2], label=r"$\dot q_1$")
    ax[1].plot(T, X[:, 3], label=r"$\dot q_2$")
    ax[1].set_ylabel("velocity [rad/s]")
    for j, c in enumerate(["C0", "C1"]):
        ax[2].plot(T, U[:, j], c, label=f"$u_{j+1}$")
        for s in (1, -1):
            ax[2].axhline(s * cfg.TORQUE_LIMIT[j], color=c, ls=":", lw=0.8)
    ax[2].set_ylabel("torque [N m]")
    ax[2].set_xlabel("time [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay(runs, title, path, x_star=cfg.X_STAR):
    """Several runs on one figure: angle-error norm and both torques.
    runs = [(label, T, X, U), ...]"""
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, (label, T, X, U) in enumerate(runs):
        T, X, U = _trim(T, X, U)
        dX = np.array([state_error(x, x_star) for x in X])
        c = f"C{i}"
        ax[0].semilogy(T, np.linalg.norm(dX[:, :2], axis=1) + 1e-6, c, label=label)
        ax[1].plot(T, U[:, 0], c)
        ax[2].plot(T, U[:, 1], c)
    ax[0].axhline(cfg.ANGLE_TOL, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel(r"$\|q-q^\star\|$ [rad]")
    for k, j in ((1, 0), (2, 1)):
        for s in (1, -1):
            ax[k].axhline(s * cfg.TORQUE_LIMIT[j], color="k", ls=":", lw=0.8)
        ax[k].set_ylabel(f"$u_{j+1}$ [N m]")
    ax[2].set_xlabel("time [s]")
    ax[0].legend(fontsize=8, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------ animation
def save_animation(T, X, U, path, title="", fps=30, x_star=cfg.X_STAR):
    """Pendulum animation with a trail and torque bars, saved as GIF/MP4."""
    T, X, U = _trim(T, X, U)
    l1, l2 = cfg.LENGTH
    step = max(1, int(round(1.0 / (fps * (T[1] - T[0])))))
    idx = np.arange(0, len(T), step)

    def joints(q):
        p1 = np.array([l1 * np.sin(q[0]), -l1 * np.cos(q[0])])
        p2 = p1 + np.array([l2 * np.sin(q[0] + q[1]), -l2 * np.cos(q[0] + q[1])])
        return p1, p2

    fig = plt.figure(figsize=(8, 4.2))
    axp = fig.add_axes([0.03, 0.08, 0.5, 0.84])
    axu = fig.add_axes([0.62, 0.15, 0.34, 0.7])
    L = 1.15 * (l1 + l2)
    axp.set_xlim(-L, L); axp.set_ylim(-L, L); axp.set_aspect("equal")
    axp.axis("off")
    g1, g2 = joints(x_star[:2])
    axp.plot([0, g1[0], g2[0]], [0, g1[1], g2[1]], color="0.8", lw=4, zorder=0)
    arm, = axp.plot([], [], "o-", lw=3, ms=7, color="C0")
    trail, = axp.plot([], [], "-", lw=1, color="C1", alpha=0.6)
    txt = axp.text(-L, L * 0.9, "", fontsize=9)
    axp.set_title(title, fontsize=10)
    lim = np.asarray(cfg.TORQUE_LIMIT)
    bars = axu.bar(["$u_1/u_{1,max}$", "$u_2/u_{2,max}$"], [0, 0], color=["C0", "C1"])
    axu.set_ylim(-1.05, 1.05); axu.axhline(0, color="k", lw=0.5)
    axu.set_title("normalised torque", fontsize=10)
    tip_hist = []

    def update(k):
        i = idx[k]
        p1, p2 = joints(X[i, :2])
        arm.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])
        tip_hist.append(p2)
        th = np.array(tip_hist[-60:])
        trail.set_data(th[:, 0], th[:, 1])
        txt.set_text(f"t = {T[i]:.2f} s")
        for b, v in zip(bars, U[i] / lim):
            b.set_height(v)
            b.set_color("C3" if abs(v) > 0.999 else b.get_facecolor())
        return arm, trail, txt, *bars

    anim = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
    writer = (animation.PillowWriter(fps=fps) if str(path).endswith(".gif")
              else animation.FFMpegWriter(fps=fps))
    anim.save(path, writer=writer)
    plt.close(fig)


# ------------------------------------------------ angle conventions (report)
def draw_conventions(path):
    """Sketch of the DFKI angle conventions and the chosen target."""
    l1, l2 = cfg.LENGTH
    q1, q2 = 0.6, 0.8
    p1 = np.array([l1 * np.sin(q1), -l1 * np.cos(q1)])
    p2 = p1 + np.array([l2 * np.sin(q1 + q2), -l2 * np.cos(q1 + q2)])
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    a = ax[0]
    a.plot([0, 0], [0, -1.1 * l1], "k--", lw=0.8)
    a.plot([p1[0], p1[0] + 0.8 * l2 * np.sin(q1)],
           [p1[1], p1[1] - 0.8 * l2 * np.cos(q1)], "k--", lw=0.8)
    a.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], "o-", lw=3, color="C0")
    t = np.linspace(0, q1, 30)
    a.plot(0.35 * l1 * np.sin(t), -0.35 * l1 * np.cos(t), "C3")
    a.text(0.42 * l1 * np.sin(q1 / 2), -0.42 * l1 * np.cos(q1 / 2), "$q_1$", color="C3")
    t = np.linspace(q1, q1 + q2, 30)
    a.plot(p1[0] + 0.4 * l2 * np.sin(t), p1[1] - 0.4 * l2 * np.cos(t), "C3")
    a.text(p1[0] + 0.5 * l2 * np.sin(q1 + q2 / 2),
           p1[1] - 0.5 * l2 * np.cos(q1 + q2 / 2), "$q_2$ (relative)", color="C3")
    a.annotate("", xy=(0, -0.1), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    a.set_title("$q_1=0$: link 1 hangs down; CCW positive")
    a.set_aspect("equal"); a.axis("off")
    b = ax[1]
    b.plot([0, 0, 0], [0, l1, l1 + l2], "o-", lw=3, color="C2")
    b.set_title(r"target $x^\star=[\pi,0,0,0]$ (upright), $u^\star=0$")
    b.set_xlim(-0.3, 0.3); b.set_aspect("equal"); b.axis("off")
    for a in ax:
        a.plot(0, 0, "ks", ms=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------ 3. initial-angle maps
def plot_map(res, title, path):
    """Success / cost / settling time over the (dq1, dq2) slice.
    res: dict from run_map (grid axis + 2-D metric arrays)."""
    g = res["grid"]
    ext = [g[0], g[-1], g[0], g[-1]]
    ok = res["success"].astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].imshow(ok, origin="lower", extent=ext,
                 cmap=ListedColormap(["#d9534f", "#5cb85c"]), vmin=0, vmax=1)
    ax[0].set_title(f"success ({100 * ok.mean():.0f}% of slice)")
    J = np.where(res["success"], res["cost_J"], np.nan)
    # robust log scale: the run starting at x* has J ~ 0 and would stretch it
    Jpos = J[np.isfinite(J) & (J > 0)]
    norm = (LogNorm(vmin=max(np.percentile(Jpos, 1), 1e-3), vmax=Jpos.max(), clip=True)
            if Jpos.size else None)
    im = ax[1].imshow(J, origin="lower", extent=ext, cmap="viridis", norm=norm)
    fig.colorbar(im, ax=ax[1], label="J (successful runs)")
    ax[1].set_title("common quadratic cost")
    ts = np.where(res["success"], res["settling_time"], np.nan)
    im = ax[2].imshow(ts, origin="lower", extent=ext, cmap="magma_r")
    fig.colorbar(im, ax=ax[2], label="s")
    ax[2].set_title("settling time")
    for a in ax:
        a.contour(g, g, ok, levels=[0.5], colors="k", linewidths=1)
        a.plot(0, 0, "w+", ms=12, mew=2)
        a.set_xlabel(r"$\Delta q_1(0)$ [rad]")
        a.set_ylabel(r"$\Delta q_2(0)$ [rad]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_map_comparison(res_a, res_b, name_a, name_b, path):
    """Who wins on the slice (figure type 3, used once Koopman exists).
    0 both fail, 1 only A, 2 only B, 3 both succeed and A cheaper, 4 both and B cheaper."""
    sa, sb = res_a["success"], res_b["success"]
    cls = np.zeros(sa.shape, int)
    cls[sa & ~sb] = 1
    cls[~sa & sb] = 2
    both = sa & sb
    cls[both & (res_a["cost_J"] <= res_b["cost_J"])] = 3
    cls[both & (res_a["cost_J"] > res_b["cost_J"])] = 4
    colors = ["#444444", "#1f77b4", "#ff7f0e", "#aec7e8", "#ffbb78"]
    labels = ["both fail", f"only {name_a}", f"only {name_b}",
              f"both, {name_a} cheaper", f"both, {name_b} cheaper"]
    g = res_a["grid"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cls, origin="lower", extent=[g[0], g[-1], g[0], g[-1]],
              cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5)
    for c, l in zip(colors, labels):
        ax.plot([], [], "s", color=c, ms=10, label=f"{l} ({np.mean(cls == labels.index(l)) * 100:.0f}%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_xlabel(r"$\Delta q_1(0)$ [rad]"); ax.set_ylabel(r"$\Delta q_2(0)$ [rad]")
    ax.set_title(f"{name_a} vs {name_b}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
