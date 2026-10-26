"""One controller class for every method in the comparison.

    u = sat( u* - K psi(x - x*) )

* local-linearization LQR : psi = identity          (K is 2 x 4)
* Koopman LQR (later)     : psi = lifting features  (K is 2 x N)

Using the same class guarantees identical angle wrapping, equilibrium
offset and saturation for all controllers.
"""
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController

from plant_utils import state_error


def identity_lift(dx):
    return dx


class LiftedLQRController(AbstractController):
    def __init__(self, K, x_star, u_star, torque_limit, lift=identity_lift, name="LQR"):
        super().__init__()
        self.K = np.asarray(K, dtype=float)
        self.x_star = np.asarray(x_star, dtype=float)
        self.u_star = np.asarray(u_star, dtype=float)
        self.torque_limit = np.asarray(torque_limit, dtype=float)
        self.lift = lift
        self.name = name

    def get_control_output_(self, x, t=None):
        dx = state_error(x, self.x_star)          # wrapped error
        z = self.lift(dx)
        u = self.u_star - self.K @ z
        return np.clip(u, -self.torque_limit, self.torque_limit)