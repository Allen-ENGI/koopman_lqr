"""Scan  training region x dictionary x gain design  on VALIDATION starts.

Three factors decide whether a Koopman LQR works at all:
  r_train : identification data restricted to |dq1|, |dq2| <= r_train
  degree  : 1 (state only, DMDc), 2 (quadratic), 3 (cubic)
  method  : "inf" infinite-horizon LQR, or "fh<T>" receding-horizon gain
            (first gain of a T-second Riccati recursion = unconstrained MPC)
The local-linearization model is scanned with the same methods (fairness:
same design choice for both models).

Selection uses random validation starts, NOT the evaluation grid.
    python scan_designs.py        # -> results/scan.csv
"""
import csv

import numpy as np

import config as cfg
import collect_data as cd
import koopman as kp
import stability as st
import fast_sweep as fs
from run_local_lqr import design_local_lqr

R_TRAIN = [0.3, 0.6, 1.0, 1.5, 2.0, np.pi]
METHODS = ["inf", "fh0.3", "fh0.5", "fh1.0"]
DICTS = [1, 2, 3, "trig1", "trig2"]
N_VAL_STARTS = 400


def val_starts(seed=cfg.SEED + 99):
    rng = np.random.default_rng(seed)
    d = rng.uniform(-np.pi, np.pi, (N_VAL_STARTS, 2))
    return cfg.X_STAR + np.column_stack([d, np.zeros((N_VAL_STARTS, 2))])


def fit_region(degree, r, train):
    Xk, Uk, Xk1 = train
    sel = (np.all(np.abs(Xk[:, :2]) <= r, axis=1)
           & np.all(np.abs(Xk1[:, :2]) <= r, axis=1))
    lift = kp.make_lift(degree, np.std(Xk[sel], axis=0))
    A, B = kp.fit(lift, Xk[sel], Uk[sel], Xk1[sel], ridge=1e-8)
    return A, B, lift, int(sel.sum())


def score(K, lift, u_star, X0):
    res = fs.run_points(K, lift, u_star, X0)
    J = res["cost_J"][res["success"]]
    return float(res["success"].mean()), float(np.median(J)) if len(J) else np.nan


def main():
    S = cd.load()
    train = cd.pairs(S["train"])
    L = design_local_lqr()
    X0 = val_starts()
    rows = []

    def record(model, r, deg, n_obs, method, K, info):
        et = st.true_local_eigs(L["Ad"], L["Bd"], K, lift)
        a_true = st.rate(et)
        # always simulate (no gating on the local rate)
        succ, J = score(K, lift, L["u_star"], X0)
        rows.append(dict(model=model, r_train=r, degree=deg, n_obs=n_obs, method=method,
                         alpha_model=st.rate(info["eig_model"]), alpha_true=a_true,
                         gain_norm=float(np.linalg.norm(K)), success=succ, median_J=J))
        print(f"{model:8s} r={r:4.2f} d={str(deg):5s} {method:6s} | alpha model {rows[-1]['alpha_model']:7.2f} "
              f"true {a_true:9.2f} | success {100 * succ:5.1f}%  medJ {J:8.2f}", flush=True)

    lift = fs.identity_lift
    for method in METHODS:
        K, P, info = st.design(L["Ad"], L["Bd"], method)
        record("local", 0.0, 1, 4, method, K, info)
    for r in R_TRAIN:
        for deg in DICTS:
            A, B, lift, n_pairs = fit_region(deg, r, train)
            for method in METHODS:
                try:
                    K, P, info = st.design(A, B, method)
                except Exception:
                    K = None
                if K is None:
                    rows.append(dict(model="koopman", r_train=r, degree=deg, n_obs=lift.n,
                                     method=method, alpha_model=np.nan, alpha_true=np.nan,
                                     gain_norm=np.nan, success=0.0, median_J=np.nan))
                    print(f"koopman  r={r:4.2f} d={str(deg):5s} {method:6s} | design failed", flush=True)
                    continue
                record("koopman", r, deg, lift.n, method, K, info)

    with open("results/scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
