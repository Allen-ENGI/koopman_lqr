"""Every controller design in the project.

  LQR         dlqr() with PBH stabilizability / detectability checks
  Local LQR   design_local_lqr() (Taylor model at x*, zero-order hold)
  Koopman LQR design() on a lifted model; true_local_eigs() checks the real
              plant; roa_certificate() samples a Lyapunov region
  Controllers LiftedLQRController  u = sat(u* - K psi(x - x*))   (balance)
              TrackingController   u = sat(u_ff(t) - K(t) e(t))  (tracking)
  Tracking    physics_controllers() (fixed-gain LQR, TVLQR), koopman_controllers()
  Factories   local_lqr_factory(), koopman_factory(name), controller_parts(name)
"""
import numpy as np
from scipy.linalg import solve_discrete_are

from double_pendulum.controller.abstract_controller import AbstractController

import config as cfg
import koopman as kp
from plant import (make_plant, equilibrium_input, linearize, c2d_zoh, wrap_angle,
                   state_error, VecPlant, design_scales)
from tool_lib import identity_lift, max_angle_error


# ======================================================================
# Discrete LQR with stabilizability / detectability checks
# ======================================================================
def pbh_stabilizable(A, B, tol=1e-8):
    """(A,B) stabilizable  <=>  rank[lam I - A, B] = n  for every |lam| >= 1."""
    n = A.shape[0]
    bad = []
    for lam in np.linalg.eigvals(A):
        if abs(lam) >= 1.0 - 1e-9:
            M = np.hstack([lam * np.eye(n) - A, B])
            if np.linalg.matrix_rank(M, tol=tol * max(1.0, np.linalg.norm(M))) < n:
                bad.append(lam)
    return len(bad) == 0, bad


def pbh_detectable(A, Cq, tol=1e-8):
    """(A, Cq) detectable  <=>  rank[lam I - A; Cq] = n  for every |lam| >= 1.
    For LQR use Cq = Q^{1/2}: every mode NOT penalized by Q must be stable."""
    ok, bad = pbh_stabilizable(A.T, Cq.T, tol)
    return ok, bad


def psd_sqrt(Q):
    """Symmetric square root of a PSD matrix that may be rank deficient
    (scipy's sqrtm returns NaNs for the lifted Q_z = C'QC)."""
    w, V = np.linalg.eigh(0.5 * (Q + Q.T))
    return V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.T


def dlqr(A, B, Q, R, check=True):
    """Gain K for v = -K z minimising sum z'Qz + v'Rv  subject to z+ = Az + Bv.

    Returns K, P (Riccati solution) and the closed-loop eigenvalues."""
    if check:
        ok_s, bad_s = pbh_stabilizable(A, B)
        Qh = psd_sqrt(Q)
        ok_d, bad_d = pbh_detectable(A, Qh)
        if not ok_s:
            raise ValueError(f"(A,B) not stabilizable; bad modes: {bad_s}")
        if not ok_d:
            raise ValueError(f"(A,Q^1/2) not detectable; bad modes: {bad_d}")
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    eig_cl = np.linalg.eigvals(A - B @ K)
    if np.max(np.abs(eig_cl)) >= 1.0:
        raise RuntimeError(f"closed loop unstable: max|eig| = {np.max(np.abs(eig_cl))}")
    return K, P, eig_cl


# ======================================================================
# Balance controller
# ======================================================================


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


# ======================================================================
# Local LQR design and controller factories
# ======================================================================
def design_local_lqr():
    """Design on the friction-free design model (see make_plant)."""
    dplant = make_plant(for_design=True)
    u_star = equilibrium_input(dplant, cfg.X_STAR)
    Ac, Bc = linearize(dplant, cfg.X_STAR, u_star, check=True)
    Ad, Bd = c2d_zoh(Ac, Bc, cfg.DT)
    K, P, eig_cl = dlqr(Ad, Bd, cfg.Q, cfg.R, check=True)
    return dict(u_star=u_star, Ac=Ac, Bc=Bc, Ad=Ad, Bd=Bd, K=K, P=P, eig_cl=eig_cl)


def local_lqr_factory():
    """Picklable factory used by tool_lib.run_map (reads cfg.TORQUE_LIMIT,
    which the condition sets)."""
    d = design_local_lqr()
    return (LiftedLQRController(d["K"], cfg.X_STAR, d["u_star"],
                                cfg.TORQUE_LIMIT, name="local LQR"), d["u_star"])


def koopman_factory(name):
    """Picklable factory for a fitted Koopman LQR (results/koopman_<name>.npz)."""
    m, lift = kp.load_model(name)
    u_star = design_local_lqr()["u_star"]
    return LiftedLQRController(m["K"], cfg.X_STAR, u_star, cfg.TORQUE_LIMIT,
                               lift=lift, name=name), u_star


def controller_parts(name):
    """(K, P, lift) for any balance controller name."""
    if name == "local":
        d = design_local_lqr()
        return d["K"], d["P"], identity_lift
    m, lift = kp.load_model(name)
    return m["K"], m["P"], lift


# ======================================================================
# Koopman LQR design and stability analysis
# ======================================================================
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


# ======================================================================
# Tracking controllers (Experiment 3)
# ======================================================================
def _wrapped_error(X):
    d = np.atleast_2d(X) - cfg.X_STAR
    d[:, :2] = wrap_angle(d[:, :2])
    return d


class TrackingController(AbstractController):
    """u = sat(u_ff[k] - K[k] (psi(dx) - psi(dx_ref[k]))),  k = round(t / dt)."""

    def __init__(self, T, X_ref, U_ff, K, torque_limit, lift=None, name=""):
        super().__init__()
        self.T = np.asarray(T)
        self.dt = float(self.T[1] - self.T[0])
        self.U_ff = np.asarray(U_ff, dtype=float)
        self.K = np.asarray(K, dtype=float)              # (2, n) or (N, 2, n)
        self.lift = lift
        self.dX_ref = _wrapped_error(X_ref)
        self.Z_ref = self.dX_ref if lift is None else lift(self.dX_ref)
        self.torque_limit = np.asarray(torque_limit, dtype=float)
        self.name = name

    def get_control_output_(self, x, t=None):
        k = int(round((0.0 if t is None else t) / self.dt))
        k = min(max(k, 0), len(self.T) - 1)
        dx = _wrapped_error(x)[0]
        if self.lift is None:
            e = dx - self.dX_ref[k]
            e[:2] = wrap_angle(e[:2])
        else:
            e = self.lift(dx) - self.Z_ref[k]
            # State block: shortest-angle error.  A letter drawn around the
            # shoulder passes hanging-down, where dq1 wraps from +pi to -pi;
            # without this the error would jump by 2 pi there.
            d = dx - self.dX_ref[k]
            d[:2] = wrap_angle(d[:2])
            e[:4] = d
        Kk = self.K[k] if self.K.ndim == 3 else self.K
        u = self.U_ff[k] - Kk @ e
        return np.clip(u, -self.torque_limit, self.torque_limit)


# ------------------------------------------------------------ physics-based
def physics_feedforward(ref):
    """Inverse dynamics of the (possibly wrong) model along the path."""
    model = VecPlant(coulomb_fric=[0.0, 0.0], mass_scale=design_scales())
    return model.inverse_dynamics(ref["X"], ref["QDD"])


def tvlqr_gains(ref, U_ff):
    """Time-varying LQR along the reference with the model's Jacobians
    (same linearization + zero-order hold as the stabilizing local LQR).
    Terminal cost: infinite-horizon LQR at the final (held) point."""
    model = make_plant(for_design=True)                  # masses x model error
    X = ref["X"]
    N = len(X)
    A, B = [], []
    for k in range(N):
        Ac, Bc = model.linear_matrices(X[k], U_ff[k])
        Ad, Bd = c2d_zoh(np.asarray(Ac, float), np.asarray(Bc, float), cfg.DT)
        A.append(Ad); B.append(Bd)
    P = solve_discrete_are(A[-1], B[-1], cfg.Q, cfg.R)
    K = np.zeros((N, 2, 4))
    for k in reversed(range(N)):
        Kk = np.linalg.solve(cfg.R + B[k].T @ P @ B[k], B[k].T @ P @ A[k])
        P = cfg.Q + A[k].T @ P @ (A[k] - B[k] @ Kk)
        K[k] = Kk
    return K


def linear_controller(ref, d):
    """FAIR local-linearization tracker: the upright Taylor model (d from
    design_local_lqr()) gives BOTH the gain and the feedforward, computed with
    the same least-squares feedforward as the Koopman trackers (psi = identity).
    Koopman vs this one = linear model vs lifted model, nothing else differs."""
    U_ff = koopman_feedforward(d["Ad"], d["Bd"], identity_lift, ref) + d["u_star"]
    return TrackingController(ref["T"], ref["X"], U_ff, d["K"], cfg.TORQUE_LIMIT,
                              name="local_linear")


def physics_controllers(ref, local_K):
    """Both physics-based trackers for one reference (under cfg.MODEL_ERROR)."""
    U_ff = physics_feedforward(ref)
    return {
        "local_fixed": TrackingController(ref["T"], ref["X"], U_ff, local_K,
                                          cfg.TORQUE_LIMIT, name="local_fixed"),
        "tvlqr": TrackingController(ref["T"], ref["X"], U_ff, tvlqr_gains(ref, U_ff),
                                    cfg.TORQUE_LIMIT, name="tvlqr"),
    }


# ------------------------------------------------------------ Koopman
def fit_tracking_models(ref, names=cfg.TRACK_OBSERVABLES):
    """Refit each observable set on data covering the letter's angles,
    then design its LQR gain.  Returns {name: (A, B, K, lift, info)}."""
    r = min(np.pi, max_angle_error(ref) + cfg.TRACK_R_MARGIN)
    S = kp.load_data()
    train = kp.pairs(S["train"])
    H = int(round(cfg.PRED_HORIZON / cfg.DT))
    val = kp.windows(S["val"], H)
    inside = kp.region_mask(val[0], r) & np.all(kp.region_mask(val[2], r), axis=1)
    val_in = kp.subset_windows(val, inside)
    out = {}
    for name in names:
        m, lift = kp.fit_dictionary(name, train, val_in, r)
        try:
            K, P, info = design(m["A"], m["B"], cfg.LQR_METHOD)
        except Exception as e:                           # Riccati failure
            K, info = None, {"error": str(e)}
        out[name] = (m["A"], m["B"], K, lift, dict(info, r_train=r))
    return out


def koopman_feedforward(A, B, lift, ref):
    """Input that makes the lifted model's predicted next STATE match the
    reference: least squares on the state rows of z+ = A z + B u."""
    Z = lift(_wrapped_error(ref["X"]))
    dX_next = _wrapped_error(ref["X"])[1:]
    target = dX_next - (Z[:-1] @ A.T)[:, :4]
    target[:, :2] = wrap_angle(target[:, :2])           # no 2 pi jump at the wrap
    U = np.linalg.lstsq(B[:4], target.T, rcond=None)[0].T
    return np.vstack([U, U[-1:]])


def koopman_controllers(ref, models):
    ctrls = {}
    for name, (A, B, K, lift, info) in models.items():
        if K is None:
            print(f"  {name}: LQR design failed ({info.get('error')}); skipped")
            continue
        U_ff = koopman_feedforward(A, B, lift, ref)
        ctrls[name] = TrackingController(ref["T"], ref["X"], U_ff, K, cfg.TORQUE_LIMIT,
                                         lift=lift, name=name)
    return ctrls




# ------------------------------------------------------------ path check
def _lift_jacobian(lift, dx, h=1e-6):
    """d psi / d dx at one point (central differences), shape (n, 4)."""
    E = h * np.eye(4)
    return ((lift(dx[None] + E) - lift(dx[None] - E)) / (2 * h)).T


def path_stability(ctrl, ref, every=10):
    """Frozen-time check along the path, the tracking version of the balance
    suitability test (an analysis check: it uses the TRUE plant's Jacobians).

    Near the reference point x_ref(t) the tracking law acts like the linear
    gain K_eff = K_x + K_extra * d psi_extra / dx (x_ref).  For state-only
    observables K_eff = K is constant; for trig and polynomial observables
    it changes along the path (cos dq, 3 dq^2, ...).  At every `every`-th
    point this returns the spectral radius of A_true - B_true K_eff.
    Frozen-time radii >= 1 do not prove failure (and < 1 does not prove
    success), but they show where the gain stops stabilizing the arm.
    Returns (% of checked points with radius >= 1, max radius)."""
    true = make_plant(for_design=False)
    dX = _wrapped_error(ref["X"])
    rho = []
    for k in range(0, len(ref["X"]), every):
        Kk = ctrl.K[k] if ctrl.K.ndim == 3 else ctrl.K
        if ctrl.lift is None:
            K_eff = Kk
        else:
            J = _lift_jacobian(ctrl.lift, dX[k])          # (n, 4), first 4 rows = I
            K_eff = Kk[:, :4] + Kk[:, 4:] @ J[4:]
        Ac, Bc = true.linear_matrices(ref["X"][k], ctrl.U_ff[k])
        Ad, Bd = c2d_zoh(np.asarray(Ac, float), np.asarray(Bc, float), cfg.DT)
        rho.append(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K_eff))))
    rho = np.asarray(rho)
    return 100.0 * float(np.mean(rho >= 1.0)), float(rho.max())