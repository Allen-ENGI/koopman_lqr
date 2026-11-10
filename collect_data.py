"""Stage 3: identification data  (x_k, u_k, x_{k+1}).

Upright is unstable, so purely random torques would only record the arm
falling.  Half of the trajectories therefore run local LQR plus random
piecewise-constant exploration torque (data near and around the target);
the other half are open-loop random piecewise-constant torques (data on the
falls and swings).  Initial angles cover the whole map slice, initial
velocities are random.

Everything is stored as WRAPPED errors dx = x - x* and du = u - u*.  A
trajectory is cut wherever a wrapped angle jumps by 2 pi (the arm passes
hanging-down) or a velocity exceeds DATA_V_MAX, so every stored segment is
continuous.  Train / validation / test are split by ORIGINAL trajectory,
never by sample.

    python collect_data.py           # -> results/ident_data.npz
"""
import numpy as np

import config as cfg
from dynamics_np import VecPlant
from plant_utils import wrap_angle


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


def load(path="results/ident_data.npz"):
    d = np.load(path)
    return {s: unpack(d[f"{s}_X"], d[f"{s}_U"], d[f"{s}_lens"])
            for s in ("train", "val", "test")}


def main():
    from run_local_lqr import design_local_lqr
    d = design_local_lqr()
    (dXc, dUc), (dXo, dUo) = generate(K=d["K"], u_star=d["u_star"])
    segs = segment(dXc, dUc)
    segs += [(pid + len(dXc), sx, su) for pid, sx, su in segment(dXo, dUo)]
    parts = split(segs, len(dXc) + len(dXo))
    out = {}
    for s, sg in parts.items():
        for k, v in pack(sg).items():
            out[f"{s}_{k}"] = v
    np.savez("results/ident_data.npz", **out)
    for s, sg in parts.items():
        n_pairs = sum(len(su) for _, su in sg)
        print(f"{s:5s}: {len(sg):5d} segments, {n_pairs:7d} pairs")
    allx = np.concatenate([sx for sx, _ in parts["train"]])
    print("train |dx| 99th pct:", np.percentile(np.abs(allx), 99, axis=0).round(2))


if __name__ == "__main__":
    main()
