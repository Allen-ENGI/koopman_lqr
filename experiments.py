"""Experimental conditions shared by BOTH simulations.

What is varied to compare local LQR with Koopman LQR (values in config.py,
section 1):
  Experiment 1  initial angles (dq1, dq2)   swept on a grid INSIDE every condition
  Experiment 2  observables                 one Koopman controller per set
  Robustness    torque limit, initial velocity
                                            one condition per value, ONE changed
                                            at a time, the other at its main value

A Condition fixes the torque limit and the initial velocity.  For the same
condition, run_local_lqr.py and run_koopman.py simulate the same plant from
the same initial states; only the controller differs.  Results are saved as
results/maps/<profile>-<realism>/<controller>__<condition>.npz.
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

import config as cfg

BASE_TORQUE = tuple(cfg.TORQUE_LIMIT)          # main torque limit of this profile
TAG = f"{cfg.PLANT_PROFILE}-{cfg.REALISM}"
SENSING = "ideal sensing" if cfg.REALISM == "ideal" else "realistic sensing"


def _fmt(values):
    return ", ".join(f"{v:g}" for v in values)


@dataclass(frozen=True)
class Condition:
    group: str                                  # "main" | "torque" | "velocity"
    torque_scale: float = 1.0
    v0: tuple = tuple(cfg.INIT_VELOCITY_MAIN)

    @property
    def torque_limit(self):
        return [round(self.torque_scale * t, 4) for t in BASE_TORQUE]

    @property
    def name(self):
        if self.group == "main":
            return "main"
        if self.group == "torque":
            return f"torque_x{self.torque_scale:.2f}"
        return f"vel_{self.v0[0]:+.1f}_{self.v0[1]:+.1f}"

    @property
    def simulator(self):
        sim = cfg.MAIN_SIMULATOR if self.group == "main" else cfg.ROBUSTNESS_SIMULATOR
        if sim == "fast" and cfg.REALISM != "ideal":
            return "dfki"                       # the fast copy has no noise / delay
        return sim

    def describe(self):
        """The fixed values of this condition, for figure titles."""
        return (f"torque limit [{_fmt(self.torque_limit)}] N·m, "
                f"initial velocity ({_fmt(self.v0)}) rad/s, {SENSING}")

    def label(self):
        """Short name of what this condition changes, for legends / tables."""
        if self.group == "main":
            return "main"
        if self.group == "torque":
            return f"torque ×{self.torque_scale:g} [{_fmt(self.torque_limit)}] N·m"
        return f"velocity ({_fmt(self.v0)}) rad/s"


MAIN = Condition("main")


def torque_conditions():
    return [Condition("torque", s) for s in cfg.TORQUE_SCALE_SWEEP]


def velocity_conditions():
    return [Condition("velocity", 1.0, tuple(float(x) for x in v))
            for v in cfg.INIT_VELOCITY_SWEEP]


def conditions(which="all"):
    """which: "main" | "robust" | "all"."""
    robust = torque_conditions() + velocity_conditions()
    return {"main": [MAIN], "robust": robust, "all": [MAIN] + robust}[which]


def apply(cond):
    """Set this condition's torque limit (plant, controllers and metrics
    all read cfg.TORQUE_LIMIT when they are built)."""
    cfg.TORQUE_LIMIT = list(cond.torque_limit)


@contextmanager
def applied(cond):
    old = list(cfg.TORQUE_LIMIT)
    apply(cond)
    try:
        yield cond
    finally:
        cfg.TORQUE_LIMIT = old


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
             torque_limit=np.array(cond.torque_limit), v0=np.array(cond.v0))
    return path


def load_map(controller, cond):
    path = map_path(controller, cond)
    if not os.path.exists(path):
        hint = ("python run_local_lqr.py --maps all" if controller == "local"
                else "python run_koopman.py --stage maps")
        raise FileNotFoundError(f"{path} is missing; run: {hint}")
    return dict(np.load(path))
