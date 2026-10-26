"""Initial-angle sweep (Experiment 1): velocities 0, sweep dq1 x dq2.

Generic over the controller: pass a top-level, picklable factory
    make_controller() -> (controller, u_star)
Each grid cell uses its own fixed noise seed, so every controller sees
exactly the same initial states AND the same noise realisation.
"""
import os
from multiprocessing import Pool

import numpy as np

import config as cfg
from plant_utils import make_plant, make_simulator
from metrics import evaluate

KEYS = ["success", "settling_time", "rmse_q", "cost_J", "effort", "sat_frac"]
_W = {}


def _init(factory):
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


def run_map(factory, n=cfg.MAP_N, rng=cfg.MAP_RANGE, workers=None):
    g = np.linspace(-rng, rng, n)
    jobs = []
    for i, d2 in enumerate(g):          # row i  -> dq2
        for j, d1 in enumerate(g):      # col j  -> dq1
            jobs.append((i * n + j, cfg.X_STAR + np.array([d1, d2, 0.0, 0.0])))
    out = {key: np.zeros((n, n)) for key in KEYS}
    with Pool(workers or os.cpu_count(), initializer=_init, initargs=(factory,)) as p:
        for k, vals in p.imap_unordered(_run_cell, jobs, chunksize=4):
            for key, v in zip(KEYS, vals):
                out[key][k // n, k % n] = v
    out["success"] = out["success"].astype(bool)
    out["grid"] = g
    return out