"""Plant construction, equilibrium, linearization and discretization."""
import numpy as np
from scipy.linalg import expm

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator

import config as cfg


def make_plant(for_design=False):
    """The one nonlinear plant used for every experiment.

    for_design=True drops Coulomb friction.  DFKI smooths it as
    cf*atan(100*qd), whose slope at qd = 0 is 100*cf, so a Jacobian would
    contain a fictitious huge damper.  The design model treats Coulomb
    friction as an unmodelled disturbance instead."""
    return SymbolicDoublePendulum(
        mass=cfg.MASS,
        length=cfg.LENGTH,
        com=cfg.COM,
        inertia=cfg.INERTIA,
        damping=cfg.DAMPING,
        coulomb_fric=[0.0, 0.0] if for_design else cfg.COULOMB_FRIC,
        gravity=cfg.GRAVITY,
        motor_inertia=cfg.MOTOR_INERTIA,
        gear_ratio=cfg.GEAR_RATIO,
        # set on the PLANT so the simulator clips every controller equally
        torque_limit=cfg.TORQUE_LIMIT,
    )


def make_simulator(plant, seed=cfg.SEED):
    """Simulator with the non-idealities selected by cfg.REALISM."""
    np.random.seed(seed)        # DFKI draws its noise from np.random
    sim = Simulator(plant=plant)
    sim.set_measurement_parameters(
        meas_noise_sigmas=cfg.MEAS_NOISE,
        delay=cfg.MEAS_DELAY,
        delay_mode="posvel" if cfg.MEAS_DELAY > 0 else "None",
    )
    sim.set_motor_parameters(u_noise_sigmas=cfg.TORQUE_NOISE,
                             u_responsiveness=cfg.MOTOR_RESPONSIVENESS)
    return sim


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return np.pi - np.mod(np.pi - np.asarray(a), 2.0 * np.pi)


def state_error(x, x_star=cfg.X_STAR):
    """dx = x - x*, with both angle errors wrapped to (-pi, pi]."""
    dx = np.asarray(x, dtype=float) - x_star
    dx[:2] = wrap_angle(dx[:2])
    return dx


def equilibrium_input(plant, x_star=cfg.X_STAR):
    """u* such that f(x*, u*) = 0.  The dynamics are affine in u, so
    f(x*, u) = f(x*, 0) + B u  and  u* solves  B u = -f(x*, 0)."""
    f0 = plant.rhs(0.0, x_star, np.zeros(2))
    _, B = plant.linear_matrices(x_star, np.zeros(2))
    u_star, *_ = np.linalg.lstsq(B, -f0, rcond=None)
    resid = plant.rhs(0.0, x_star, u_star)
    if np.max(np.abs(resid)) > 1e-9:
        raise ValueError(f"x* is not an equilibrium: f(x*,u*) = {resid}")
    if np.any(np.abs(u_star) > np.asarray(cfg.TORQUE_LIMIT)):
        raise ValueError(f"u* = {u_star} exceeds the torque limit")
    return u_star


def linearize(plant, x_star, u_star, check=True, eps=1e-6):
    """Continuous-time Jacobians A_c, B_c of xdot = f(x,u) at (x*,u*).

    Uses DFKI's symbolic Jacobian, cross-checked against central finite
    differences of plant.rhs (the function the simulator integrates)."""
    Ac, Bc = plant.linear_matrices(x_star, u_star)
    if check:
        Afd = np.zeros((4, 4))
        Bfd = np.zeros((4, 2))
        for i in range(4):
            e = np.zeros(4); e[i] = eps
            Afd[:, i] = (plant.rhs(0, x_star + e, u_star)
                         - plant.rhs(0, x_star - e, u_star)) / (2 * eps)
        for j in range(2):
            e = np.zeros(2); e[j] = eps
            Bfd[:, j] = (plant.rhs(0, x_star, u_star + e)
                         - plant.rhs(0, x_star, u_star - e)) / (2 * eps)
        errA = np.max(np.abs(Ac - Afd))
        errB = np.max(np.abs(Bc - Bfd))
        if max(errA, errB) > 1e-5:
            raise RuntimeError(f"Jacobian mismatch: |dA|={errA:.2e}, |dB|={errB:.2e}")
    return Ac, Bc


def c2d_zoh(Ac, Bc, dt):
    """Exact zero-order-hold discretization (matches the simulator,
    which holds u constant over each step).  Not DFKI's
    linear_matrices_discrete(), which is forward Euler."""
    n, m = Bc.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]