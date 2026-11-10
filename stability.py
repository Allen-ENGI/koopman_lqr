"""Stability analysis shared by every controller.

1. Koopman LQR design on a lifted model (with stabilizability / detectability).
2. Local exponential stability on the TRUE plant: for u = u* - K psi(dx),
   d(K psi)/d dx at 0 = K[:, :4]  (higher monomials have zero slope), so the
   true local closed loop is  A_d - B_d K[:, :4]  with the Taylor A_d, B_d.
3. Continuous-time rates  alpha = -ln max|lambda| / dt.
4. Sampled Lyapunov region-of-attraction certificate:
   largest rho such that V(f_cl(x)) < V(x) for every sample with V(x) <= rho,
   where V = psi(dx)' P psi(dx) comes from the controller's own Riccati
   solution and f_cl is the true saturated one-step map.
   Sampled, not proven (an SOS program would make it rigorous).
"""
import numpy as np

import config as cfg
from lqr_design import dlqr, pbh_stabilizable, pbh_detectable, psd_sqrt
from dynamics_np import VecPlant
from plant_utils import wrap_angle


def lifted_cost(n):
    Qz = np.zeros((n, n)); Qz[:4, :4] = cfg.Q
    return Qz


def design_koopman_lqr(A, B):
    """LQR on z+ = A z + B v with Q_z = C'QC.  Adds LIFTED_Q_EPS on the
    lifted states only if (A, Q_z^1/2) is not detectable."""
    n = A.shape[0]
    Qz = lifted_cost(n)
    info = dict(stabilizable=pbh_stabilizable(A, B)[0], regularized=False)
    if not info["stabilizable"]:
        return None, None, info
    det_ok = pbh_detectable(A, psd_sqrt(Qz))[0]
    info["detectable"] = det_ok
    if not det_ok:
        Qz[4:, 4:] += cfg.LIFTED_Q_EPS * np.eye(n - 4)
        info["regularized"] = True
    K, P, eig_model = dlqr(A, B, Qz, cfg.R, check=True)
    info["eig_model"] = eig_model
    return K, P, info


def finite_horizon_gain(A, B, Qz, R, horizon_s, dt=cfg.DT):
    """First gain of the N-step Riccati recursion with terminal cost Qz:
    the static gain of unconstrained receding-horizon (MPC) control.
    Returns K and the cost-to-go matrix P_0 (used as Lyapunov candidate)."""
    P = Qz.copy()
    for _ in range(int(round(horizon_s / dt))):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Qz + A.T @ P @ (A - B @ K)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def design(A, B, method):
    """method = "inf" (infinite-horizon LQR) or "fh<seconds>" (receding)."""
    n = A.shape[0]
    if method == "inf":
        if n == 4:
            K, P, eig = dlqr(A, B, cfg.Q, cfg.R, check=True)
            return K, P, dict(eig_model=eig, regularized=False)
        return design_koopman_lqr(A, B)
    K, P = finite_horizon_gain(A, B, lifted_cost(n), cfg.R, float(method[2:]))
    return K, P, dict(eig_model=np.linalg.eigvals(A - B @ K), regularized=False)


def rate(eigs, dt=cfg.DT):
    return float(-np.log(np.max(np.abs(eigs))) / dt)


def local_gain(K, lift=None):
    """Effective linear gain of u = -K psi(dx) at the target:
    K * d psi/d dx (0).  Equals K[:, :4] for polynomial lifts."""
    if lift is None or not hasattr(lift, "jacobian0"):
        return K[:, :4]
    return K @ lift.jacobian0()


def true_local_eigs(Ad, Bd, K, lift=None):
    """Closed-loop eigenvalues of the TRUE plant's linearization under the
    controller u = u* - K psi(dx)."""
    return np.linalg.eigvals(Ad - Bd @ local_gain(K, lift))


# ------------------------------------------------ sampled ROA certificate
def _closed_loop_step(vp, X, K, lift, u_star):
    dX = X - cfg.X_STAR
    dX[:, :2] = wrap_angle(dX[:, :2])
    U = vp.sat(u_star - lift(dX) @ K.T)
    return vp.rk4_step(X, U)


def _V(P, lift, dX):
    Z = lift(dX)
    return np.einsum("ij,jk,ik->i", Z, P, Z)


def roa_certificate(K, P, lift, u_star, n=cfg.ROA_SAMPLES, chunk=200_000,
                    seed=cfg.SEED):
    """Return rho* and diagnostics.  Samples uniformly in the 4-D box
    angles in [-pi, pi], velocities in [-ROA_V_BOX, ROA_V_BOX]."""
    rng = np.random.default_rng(seed + 11)
    vp = VecPlant()
    hi = np.array([np.pi, np.pi, cfg.ROA_V_BOX, cfg.ROA_V_BOX])
    rho_viol = np.inf
    for s in range(0, n, chunk):
        dX = rng.uniform(-hi, hi, size=(min(chunk, n - s), 4))
        V0 = _V(P, lift, dX)
        X1 = _closed_loop_step(vp, cfg.X_STAR + dX, K, lift, u_star)
        dX1 = X1 - cfg.X_STAR
        dX1[:, :2] = wrap_angle(dX1[:, :2])
        V1 = _V(P, lift, dX1)
        viol = V1 >= V0
        if np.any(viol):
            rho_viol = min(rho_viol, float(V0[viol].min()))
    # the certified set must also lie inside the sampled box:
    # min of V over the box faces (dense random points on each face)
    face = []
    for i in range(4):
        for sgn in (-1, 1):
            F = rng.uniform(-hi, hi, size=(50_000, 4)); F[:, i] = sgn * hi[i]
            face.append(_V(P, lift, F).min())
    rho_box = float(min(face))
    return dict(rho=min(rho_viol, rho_box), rho_viol=rho_viol, rho_box=rho_box)


def certified_slice(P, lift, rho, grid):
    """V on the zero-velocity (dq1, dq2) slice, and the certified mask."""
    D1, D2 = np.meshgrid(grid, grid)
    dX = np.column_stack([D1.ravel(), D2.ravel(), np.zeros(D1.size), np.zeros(D1.size)])
    V = _V(P, lift, dX).reshape(D1.shape)
    return V, V <= rho
