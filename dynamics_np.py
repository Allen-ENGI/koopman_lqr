"""Vectorized copy of DFKI's double-pendulum dynamics ("UnderactuatedLecture").

Used where DFKI's one-state-at-a-time `plant.rhs` would be too slow:
batch data generation and million-sample region-of-attraction checks.
`check_against_dfki()` verifies it matches DFKI to machine precision, so it
is the SAME plant, not a second model.  The closed-loop maps still run in
DFKI's own Simulator.

    M(q) qdd + C(q, qd) qd - G(q) + F(qd) = u        (DFKI sign convention)
"""
import numpy as np

import config as cfg


class VecPlant:
    def __init__(self, coulomb_fric=None):
        self.m1, self.m2 = cfg.MASS
        self.l1, self.l2 = cfg.LENGTH
        self.r1, self.r2 = cfg.COM
        self.I1, self.I2 = cfg.INERTIA
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


def check_against_dfki(n=200, seed=1):
    """Max abs difference to DFKI's plant.rhs at random states/inputs."""
    from plant_utils import make_plant
    rng = np.random.default_rng(seed)
    plant, vp = make_plant(), VecPlant()
    x = rng.uniform([-4, -4, -10, -10], [4, 4, 10, 10], size=(n, 4))
    u = rng.uniform(-3, 3, size=(n, 2))
    ref = np.array([plant.rhs(0.0, xi, ui) for xi, ui in zip(x, u)])
    return float(np.max(np.abs(ref - vp.rhs(x, u))))


if __name__ == "__main__":
    print("max |rhs_vec - rhs_DFKI| =", check_against_dfki())
