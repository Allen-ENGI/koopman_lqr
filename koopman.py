"""Koopman models: identification data and lifted linear predictors.

  Data        generate(), segment(), split(), load_data(), pairs()
              (whole trajectories, cut at angle wraps, split by trajectory)
  Observables PolyLift (poly1 state only, poly2 quadratic, poly3 cubic, ...),
              TrigLift (trig1, trig2), make_lift()
  Fitting     fit() (ridge least squares for [A B]), fit_dictionary()
  Prediction  predict(), multistep_rmse(), windows()

Predictor (Korda & Mezic 2018):  z = psi(dx) = [dx ; observables],
z+ = A z + B du,  dx = C z with C = [I 0],  psi(0) = 0.
"""
from itertools import combinations_with_replacement

import numpy as np

import config as cfg
from plant import VecPlant, wrap_angle


# ======================================================================
# Identification data
# ======================================================================
def _piecewise_constant(rng, n_traj, n_steps, amp):
    """Random torques held for random durations in DATA_HOLD."""
    U = np.zeros((n_traj, n_steps, 2))
    lo, hi = (int(round(h / cfg.DT)) for h in cfg.DATA_HOLD)
    for i in range(n_traj):
        k = 0
        while k < n_steps:
            L = rng.integers(lo, hi + 1)
            U[i, k:k + L] = rng.uniform(-amp, amp)
            k += L
    return U


def _rollout(rng, vp, n, T, closed, K, u_star):
    steps = int(round(T / cfg.DT))
    lim = np.asarray(cfg.TORQUE_LIMIT)
    dx0 = np.column_stack([rng.uniform(-np.pi, np.pi, (n, 2)),
                           rng.uniform(-cfg.DATA_V0, cfg.DATA_V0, (n, 2))])
    x = cfg.X_STAR + dx0
    W = _piecewise_constant(rng, n, steps, (0.5 if closed else 1.0) * lim)
    X = np.zeros((n, steps + 1, 4)); U = np.zeros((n, steps, 2))
    X[:, 0] = x
    for k in range(steps):
        if closed:
            dx = x - cfg.X_STAR
            dx[:, :2] = wrap_angle(dx[:, :2])
            u = vp.sat(u_star - dx @ K.T + W[:, k])
        else:
            u = vp.sat(W[:, k])
        x = vp.rk4_step(x, u)
        X[:, k + 1] = x; U[:, k] = u
    dX = X - cfg.X_STAR
    dX[..., :2] = wrap_angle(dX[..., :2])
    return dX, U - u_star


def generate(seed=cfg.SEED, K=None, u_star=None):
    """Returns two lists of trajectories (closed-loop, open-loop)."""
    rng = np.random.default_rng(seed)
    vp = VecPlant()
    c = _rollout(rng, vp, cfg.DATA_N_CLOSED, cfg.DATA_T_CLOSED, True, K, u_star)
    o = _rollout(rng, vp, cfg.DATA_N_OPEN, cfg.DATA_T_OPEN, False, K, u_star)
    return c, o


def segment(dX, dU, min_len=20):
    """Cut trajectories at wrap jumps / overspeed.  Returns list of
    (parent_id, dX_seg (L+1,4), dU_seg (L,2))."""
    jump = np.any(np.abs(np.diff(dX[..., :2], axis=1)) > np.pi, axis=2)
    fast = np.any(np.abs(dX[..., 2:]) > cfg.DATA_V_MAX, axis=2)
    ok = ~jump & ~fast[:, :-1] & ~fast[:, 1:]               # (n, steps) per pair
    segs = []
    for i in range(dX.shape[0]):
        k = 0
        while k < ok.shape[1]:
            if not ok[i, k]:
                k += 1; continue
            j = k
            while j < ok.shape[1] and ok[i, j]:
                j += 1
            if j - k >= min_len:
                segs.append((i, dX[i, k:j + 1], dU[i, k:j]))
            k = j
    return segs


def split(segs, n_traj, seed=cfg.SEED):
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n_traj)
    a = int(cfg.DATA_SPLIT[0] * n_traj); b = a + int(cfg.DATA_SPLIT[1] * n_traj)
    which = np.empty(n_traj, dtype="<U5")
    which[perm[:a]], which[perm[a:b]], which[perm[b:]] = "train", "val", "test"
    out = {"train": [], "val": [], "test": []}
    for pid, sx, su in segs:
        out[which[pid]].append((sx, su))
    return out


def pack(segs):
    """List of segments -> flat arrays + offsets (npz friendly)."""
    lens = np.array([len(su) for _, su in segs])
    return dict(X=np.concatenate([sx for sx, _ in segs]),
                U=np.concatenate([su for _, su in segs]),
                lens=lens)


def unpack(X, U, lens):
    segs, ix, iu = [], 0, 0
    for L in lens:
        segs.append((X[ix:ix + L + 1], U[iu:iu + L]))
        ix += L + 1; iu += L
    return segs


def pairs(segs):
    """Segments -> one-step pairs (dx_k, du_k, dx_{k+1})."""
    return (np.concatenate([sx[:-1] for sx, _ in segs]),
            np.concatenate([su for _, su in segs]),
            np.concatenate([sx[1:] for sx, _ in segs]))


def load_data(path="results/ident_data.npz"):
    d = np.load(path)
    return {s: unpack(d[f"{s}_X"], d[f"{s}_U"], d[f"{s}_lens"])
            for s in ("train", "val", "test")}


# ======================================================================
# Observables, fitting, prediction
# ======================================================================
STATE_NAMES = ["dq1", "dq2", "dqd1", "dqd2"]


class PolyLift:
    """Picklable lifting function (used inside the controller)."""

    def __init__(self, degree, sigma):
        self.degree = int(degree)
        self.sigma = np.asarray(sigma, dtype=float)
        self.terms = [c for d in range(2, self.degree + 1)
                      for c in combinations_with_replacement(range(4), d)]
        # index matrix padded with a column that points at a constant 1
        width = max([len(t) for t in self.terms], default=1)
        self._idx = np.full((len(self.terms), width), 4, dtype=int)
        for r, t in enumerate(self.terms):
            self._idx[r, :len(t)] = t

    @property
    def n(self):
        return 4 + len(self.terms)

    def names(self):
        return STATE_NAMES + ["*".join(f"s{i + 1}" for i in t) for t in self.terms]

    def __call__(self, dx):
        dx = np.asarray(dx, dtype=float)
        single = dx.ndim == 1
        X = dx[None] if single else dx
        if self.terms:
            S = np.concatenate([X / self.sigma, np.ones((len(X), 1))], axis=1)
            Z = np.concatenate([X, np.prod(S[:, self._idx], axis=2)], axis=1)
        else:
            Z = X
        return Z[0] if single else Z

    def jacobian0(self):
        """d psi / d dx at 0: [I; 0] (higher monomials have zero slope)."""
        J = np.zeros((self.n, 4)); J[:4] = np.eye(4)
        return J


class TrigLift:
    """Physics-matched dictionary (extension of Experiment 2).

    Relative to upright, sin(q1) = -sin(dq1) and sin(q1+q2) = -sin(dq1+dq2),
    so gravity is LINEAR in these observables; the Coriolis terms are linear
    in sin(dq2)*qd_i*qd_j.  All are zero at the target (psi(0) = 0) and
    bounded, so they cannot produce polynomial-type runaway lifted modes.
      level 1: [dx, sin dq1, sin(dq1+dq2), sin dq2, 1-cos dq2]          (8)
      level 2: level 1 + sin dq2 * [qd1^2, qd1 qd2, qd2^2]
                       + (1-cos dq2) * [dq1, dq2]                         (13)
    """

    def __init__(self, level=1, sigma=None):
        self.level = int(level)
        self.degree = f"trig{self.level}"
        self.sigma = np.ones(4) if sigma is None else np.asarray(sigma, float)

    @property
    def n(self):
        return 8 if self.level == 1 else 13

    def names(self):
        nm = STATE_NAMES + ["sin dq1", "sin(dq1+dq2)", "sin dq2", "1-cos dq2"]
        if self.level >= 2:
            nm += ["sin dq2 qd1^2", "sin dq2 qd1 qd2", "sin dq2 qd2^2",
                   "(1-cos dq2) dq1", "(1-cos dq2) dq2"]
        return nm

    def __call__(self, dx):
        dx = np.asarray(dx, dtype=float)
        single = dx.ndim == 1
        X = dx[None] if single else dx
        q1, q2, w1, w2 = X.T
        s2, c2 = np.sin(q2), 1.0 - np.cos(q2)
        cols = [X, np.column_stack([np.sin(q1), np.sin(q1 + q2), s2, c2])]
        if self.level >= 2:
            cols.append(np.column_stack([s2 * w1 * w1, s2 * w1 * w2, s2 * w2 * w2,
                                         c2 * q1, c2 * q2]))
        Z = np.concatenate(cols, axis=1)
        return Z[0] if single else Z

    def jacobian0(self):
        """d psi / d dx at 0.  sin terms have slope 1, (1-cos) and all
        level-2 products have zero slope."""
        J = np.zeros((self.n, 4)); J[:4] = np.eye(4)
        J[4, 0] = 1.0                 # sin dq1
        J[5, 0] = J[5, 1] = 1.0       # sin(dq1 + dq2)
        J[6, 1] = 1.0                 # sin dq2
        return J


def make_lift(kind, sigma):
    """kind: int degree (polynomial) or "trig1" / "trig2"."""
    if isinstance(kind, str) and kind.startswith("trig"):
        return TrigLift(int(kind[4:]), sigma)
    if isinstance(kind, str) and kind.startswith("poly"):
        return PolyLift(int(kind[4:]), sigma)
    return PolyLift(int(kind), sigma)


def fit(lift, Xk, Uk, Xk1, ridge=0.0):
    """Least squares  min ||Z+ - A Z - B U||^2 + ridge * scale * ||[A B]||^2."""
    Z, Zp = lift(Xk), lift(Xk1)
    Phi = np.concatenate([Z, Uk], axis=1)              # (N, n + m)
    G = Phi.T @ Phi
    lam = ridge * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Phi.T @ Zp).T
    n = lift.n
    return W[:, :n], W[:, n:]


def windows(segs, H, max_windows=4000, seed=cfg.SEED):
    """Random length-H windows (x0, u_0..u_{H-1}, x_1..x_H) from segments."""
    rng = np.random.default_rng(seed + 7)
    cand = [(i, k) for i, (_, su) in enumerate(segs)
            for k in range(0, len(su) - H + 1, max(1, H // 4))]
    sel = rng.choice(len(cand), size=min(max_windows, len(cand)), replace=False)
    X0 = np.array([segs[cand[s][0]][0][cand[s][1]] for s in sel])
    Useq = np.array([segs[cand[s][0]][1][cand[s][1]:cand[s][1] + H] for s in sel])
    Xtrue = np.array([segs[cand[s][0]][0][cand[s][1] + 1:cand[s][1] + H + 1] for s in sel])
    return X0, Useq, Xtrue


def predict(A, B, lift, X0, Useq):
    """Pure linear propagation in the lifted space (no re-lifting)."""
    z = lift(X0)
    out = np.zeros(Useq.shape[:2] + (4,))
    with np.errstate(all="ignore"):
        for k in range(Useq.shape[1]):
            z = z @ A.T + Useq[:, k] @ B.T
            out[:, k] = z[:, :4]
    return out


def multistep_rmse(A, B, lift, win):
    """Joint-angle RMSE (rad) at each prediction step, over all windows."""
    X0, Useq, Xtrue = win
    P = predict(A, B, lift, X0, Useq)
    err = np.sum((P[..., :2] - Xtrue[..., :2]) ** 2, axis=2)    # (W, H)
    with np.errstate(all="ignore"):
        r = np.sqrt(np.mean(err, axis=0))
    r[~np.isfinite(r)] = np.inf
    return r


def region_mask(X, r):
    """Samples whose two angle errors are within +-r."""
    return np.all(np.abs(X[..., :2]) <= r, axis=-1)


def fit_dictionary(name, train_pairs, val_win, r=cfg.KOOPMAN_R_TRAIN):
    """Fit one dictionary on training pairs inside the region; choose the
    ridge weight on validation multistep error (windows inside the region)."""
    Xk, Uk, Xk1 = train_pairs
    sel = region_mask(Xk, r) & region_mask(Xk1, r)
    lift = make_lift(name, np.std(Xk[sel], axis=0))
    H = int(round(cfg.PRED_HORIZON / cfg.DT))
    best = None
    for ridge in cfg.RIDGE_GRID:
        A, B = fit(lift, Xk[sel], Uk[sel], Xk1[sel], ridge)
        score = float(np.mean(multistep_rmse(A, B, lift, val_win)[:H]))
        if best is None or score < best[0]:
            best = (score, ridge, A, B)
    score, ridge, A, B = best
    return dict(name=name, sigma=lift.sigma, A=A, B=B, ridge=ridge, r_train=r,
                val_score=score, n_obs=lift.n, n_pairs=int(sel.sum())), lift


def subset_windows(win, mask):
    return tuple(w[mask] for w in win)


def load_model(name, folder="results"):
    m = dict(np.load(f"{folder}/koopman_{name}.npz", allow_pickle=True))
    return m, make_lift(name, m["sigma"])
