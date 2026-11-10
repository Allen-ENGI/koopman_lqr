"""Initial-angle maps: one controller, every initial state of one condition.

    run_map(factory, cond)

`factory()` returns (controller, u_star) and must be a top-level function
(or functools.partial of one) so worker processes can rebuild it.
`cond` (experiments.Condition) fixes the torque limit and initial velocity;
the initial angles are swept on the config.MAP_N grid.  Each grid cell uses
its own fixed noise seed, so every controller sees exactly the same initial
states AND the same noise.
"""
import os
from multiprocessing import Pool

import numpy as np

import config as cfg
import experiments as ex
from plant_utils import make_plant, make_simulator
from metrics import evaluate

KEYS = ["success", "settling_time", "rmse_q", "cost_J", "effort", "sat_frac",
        "alpha_hat", "c_hat"]
_W = {}


def _init(factory, cond):
    # Applied inside the worker: on macOS / Windows workers start fresh and
    # would otherwise silently use the main torque limit.
    ex.apply(cond)
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
    g, X0 = ex.initial_states(cond)
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
        import fast_sweep
        with ex.applied(cond):
            ctrl, u_star = factory()
            res = fast_sweep.run_grid(ctrl.K, ctrl.lift, u_star, cond)
    else:
        res = _run_map_dfki(factory, cond)
    res["simulator"] = sim
    return res
