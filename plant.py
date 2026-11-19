"""The plant: DFKI's double pendulum and everything about its dynamics.

  make_plant / make_simulator   DFKI plant and simulator (true plant or design model)
  equilibrium_input, linearize, c2d_zoh
                                equilibrium, Jacobians, zero-order-hold discretization
  VecPlant                      vectorized copy of DFKI's equations (checked to ~1e-13),
                                with inverse dynamics, used for data and fast sweeps
  wrap_angle, state_error       angle wrapping

    python plant.py     # check the vectorized copy against DFKI
"""
import numpy as np
from scipy.linalg import expm

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator

import config as cfg


# ======================================================================
# DFKI plant, simulator, linearization
# ======================================================================
def design_scales(error=None):
    """Per-link (shoulder, elbow) mass scale of the physics design model:
    cfg.MODEL_ERROR on the links named by cfg.MODEL_ERROR_LINKS."""
    e = cfg.MODEL_ERROR if error is None else float(error)
    return {"both": (e, e), "shoulder": (e, 1.0),
            "elbow": (1.0, e)}[cfg.MODEL_ERROR_LINKS]


def make_plant(for_design=False):
    """The one nonlinear plant used for every experiment.

    for_design=False: the TRUE plant (what is simulated): link masses and
                      inertias x cfg.PLANT_MASS (plant change, 1.0 = nominal).
    for_design=True:  the MODEL a physics-based design believes in, built
                      from the NOMINAL plant (designs never see plant change)
      * drops Coulomb friction.  DFKI smooths it as cf*atan(100*qd), whose
        slope at qd = 0 is 100*cf, so a Jacobian would contain a fictitious
        huge damper; the design treats it as an unmodelled disturbance.
      * scales link masses and inertias by cfg.MODEL_ERROR on the links in
        cfg.MODEL_ERROR_LINKS (1.0 = exact model).  Koopman models never
        use this: they are fitted to data from the true plant."""
    s = design_scales() if for_design else (cfg.PLANT_MASS, cfg.PLANT_MASS)
    return SymbolicDoublePendulum(
        mass=[k * m for k, m in zip(s, cfg.MASS)],
        length=cfg.LENGTH,
        com=cfg.COM,
        inertia=[k * i for k, i in zip(s, cfg.INERTIA)],
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


# ======================================================================
# Vectorized copy of DFKI's dynamics
# ======================================================================
class VecPlant:
    def __init__(self, coulomb_fric=None, mass_scale=1.0):
        """mass_scale != 1 scales link masses and inertias: one number for
        both links, or a (shoulder, elbow) pair (plant.design_scales())."""
        s1, s2 = np.broadcast_to(np.asarray(mass_scale, dtype=float), (2,))
        self.m1, self.m2 = s1 * cfg.MASS[0], s2 * cfg.MASS[1]
        self.l1, self.l2 = cfg.LENGTH
        self.r1, self.r2 = cfg.COM
        self.I1, self.I2 = s1 * cfg.INERTIA[0], s2 * cfg.INERTIA[1]
        self.b1, self.b2 = cfg.DAMPING
        cf = cfg.COULOMB_FRIC if coulomb_fric is None else coulomb_fric
        self.cf1, self.cf2 = cf
        self.g = cfg.GRAVITY
        self.Ir, self.gr = cfg.MOTOR_INERTIA, cfg.GEAR_RATIO
        self.tl = np.asarray(cfg.TORQUE_LIMIT, dtype=float)

    def accel(self, x, u):
        """x (N,4), u (N,2) -> qdd (N,2)."""
        q1, q2, qd1, qd2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        m2, l1, r2 = self.m2, self.l1, self.r2
        c2, s2 = np.cos(q2), np.sin(q2)
        M11 = (self.I1 + self.I2 + m2 * l1 ** 2 + 2 * m2 * l1 * r2 * c2
               + self.gr ** 2 * self.Ir + self.Ir)
        M12 = self.I2 + m2 * l1 * r2 * c2 - self.gr * self.Ir
        M22 = self.I2 + self.gr ** 2 * self.Ir
        h = m2 * l1 * r2 * s2
        # C qd
        Cqd1 = -2 * h * qd2 * qd1 - h * qd2 * qd2
        Cqd2 = h * qd1 * qd1
        G1 = (-self.m1 * self.g * self.r1 * np.sin(q1)
              - m2 * self.g * (l1 * np.sin(q1) + r2 * np.sin(q1 + q2)))
        G2 = -m2 * self.g * r2 * np.sin(q1 + q2)
        F1 = self.b1 * qd1 + self.cf1 * np.arctan(100 * qd1)
        F2 = self.b2 * qd2 + self.cf2 * np.arctan(100 * qd2)
        f1 = u[:, 0] - Cqd1 + G1 - F1
        f2 = u[:, 1] - Cqd2 + G2 - F2
        det = M11 * M22 - M12 * M12
        return np.column_stack([(M22 * f1 - M12 * f2) / det,
                                (-M12 * f1 + M11 * f2) / det])

    def rhs(self, x, u):
        return np.column_stack([x[:, 2], x[:, 3], self.accel(x, u)])

    def rk4_step(self, x, u, dt=cfg.DT):
        """One DFKI simulator step: RK4 with u held, then x += dt * slope."""
        k1 = self.rhs(x, u)
        k2 = self.rhs(x + 0.5 * dt * k1, u)
        k3 = self.rhs(x + 0.5 * dt * k2, u)
        k4 = self.rhs(x + dt * k3, u)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def sat(self, u):
        return np.clip(u, -self.tl, self.tl)

    def inverse_dynamics(self, x, qdd):
        """Torque that produces acceleration qdd at state x:
        u = M qdd + C qd - G + F   (x (N,4), qdd (N,2) -> u (N,2)).
        Same formulas as accel(); check with check_inverse_dynamics()."""
        q1, q2, qd1, qd2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        m2, l1, r2 = self.m2, self.l1, self.r2
        c2, s2 = np.cos(q2), np.sin(q2)
        M11 = (self.I1 + self.I2 + m2 * l1 ** 2 + 2 * m2 * l1 * r2 * c2
               + self.gr ** 2 * self.Ir + self.Ir)
        M12 = self.I2 + m2 * l1 * r2 * c2 - self.gr * self.Ir
        M22 = self.I2 + self.gr ** 2 * self.Ir
        h = m2 * l1 * r2 * s2
        Cqd1 = -2 * h * qd2 * qd1 - h * qd2 * qd2
        Cqd2 = h * qd1 * qd1
        G1 = (-self.m1 * self.g * self.r1 * np.sin(q1)
              - m2 * self.g * (l1 * np.sin(q1) + r2 * np.sin(q1 + q2)))
        G2 = -m2 * self.g * r2 * np.sin(q1 + q2)
        F1 = self.b1 * qd1 + self.cf1 * np.arctan(100 * qd1)
        F2 = self.b2 * qd2 + self.cf2 * np.arctan(100 * qd2)
        u1 = M11 * qdd[:, 0] + M12 * qdd[:, 1] + Cqd1 - G1 + F1
        u2 = M12 * qdd[:, 0] + M22 * qdd[:, 1] + Cqd2 - G2 + F2
        return np.column_stack([u1, u2])


def check_against_dfki(n=200, seed=1):
    """Max abs difference to DFKI's plant.rhs at random states/inputs."""
    rng = np.random.default_rng(seed)
    plant, vp = make_plant(), VecPlant()
    x = rng.uniform([-4, -4, -10, -10], [4, 4, 10, 10], size=(n, 4))
    u = rng.uniform(-3, 3, size=(n, 2))
    ref = np.array([plant.rhs(0.0, xi, ui) for xi, ui in zip(x, u)])
    return float(np.max(np.abs(ref - vp.rhs(x, u))))


def check_inverse_dynamics(n=200, seed=2):
    """inverse_dynamics(x, accel(x, u)) must give back u."""
    rng = np.random.default_rng(seed)
    vp = VecPlant()
    x = rng.uniform([-4, -4, -10, -10], [4, 4, 10, 10], size=(n, 4))
    u = rng.uniform(-3, 3, size=(n, 2))
    return float(np.max(np.abs(vp.inverse_dynamics(x, vp.accel(x, u)) - u)))


if __name__ == "__main__":
    print("max |rhs_vec - rhs_DFKI| =", check_against_dfki())
    print("max |ID(FD(u)) - u|      =", check_inverse_dynamics())
