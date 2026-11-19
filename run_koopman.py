"""SIMULATION 2 of 2 (balance): Koopman LQR, and the comparison with local LQR.

Design: lift the error z = psi(dx) with each observable set in
config.OBSERVABLES, fit z+ = A_z z + B_z du to data, solve the same Riccati
equation (Q_z = C'QC, same R) -> u = sat(u* - K_z psi(dx)).  Then simulate
every Koopman controller over the SAME conditions as run_local_lqr.py
(config.py, section 1) and compare.

    python run_local_lqr.py --maps all        # SIMULATION 1 first (local LQR maps)
    python run_koopman.py                     # the core stages, in order:
                                              #   data, fit, design, maps, figures
    python run_koopman.py --stage fit         # one stage: data | fit | design | maps | figures
    python run_koopman.py --stage maps --maps main     # only the main-condition maps
    python run_koopman.py --stage extra       # optional analysis, not needed for the
                                              #   comparison: design scan, Lyapunov
                                              #   certificates, convergence-rate figures

Core figures: prediction error, success maps, who-wins, success and cost by
distance from upright, time series, summary tables, robustness.

What is compared (config.py, section 1):
  Experiment 1  initial angles          -> axes of every map, and distance bands
  Experiment 2  observables             -> one Koopman controller per set
  Robustness    config.ROBUSTNESS (default: model error); one changed at a time
Fixed for all: plant, target, dt, Q, R, success test, initial-angle grid,
training data, gain design (infinite-horizon LQR).
"""
import argparse
import csv
import functools
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config as cfg
import tool_lib as tl
import koopman as kp
import control as ctl
from control import design_local_lqr
from tool_lib import identity_lift

TAG = tl.TAG
RES, FIG = "results", "figures"
FIXED_MAIN = tl.MAIN.describe()
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
tl.apply_style()


# ============================================================ controllers
def koopman_names():
    """Koopman controllers whose gain stabilizes the real plant near x*."""
    info = json.load(open(f"{RES}/design_summary.json"))
    return [n for n in cfg.OBSERVABLES if info[n]["suitable"]]


def suitable_names():
    return ["local"] + koopman_names()


# ============================================================ stage: data
def stage_data():
    """Identification data (whole trajectories, split by trajectory)."""
    d = design_local_lqr()
    (dXc, dUc), (dXo, dUo) = kp.generate(K=d["K"], u_star=d["u_star"])
    segs = kp.segment(dXc, dUc)
    segs += [(pid + len(dXc), sx, su) for pid, sx, su in kp.segment(dXo, dUo)]
    parts = kp.split(segs, len(dXc) + len(dXo))
    out = {}
    for s, sg in parts.items():
        for k, v in kp.pack(sg).items():
            out[f"{s}_{k}"] = v
    np.savez("results/ident_data.npz", **out)
    for s, sg in parts.items():
        n_pairs = sum(len(su) for _, su in sg)
        print(f"{s:5s}: {len(sg):5d} segments, {n_pairs:7d} pairs")
    allx = np.concatenate([sx for sx, _ in parts["train"]])
    print("train |dx| 99th pct:", np.percentile(np.abs(allx), 99, axis=0).round(2))


# ============================================================ stage: scan
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
    res = tl.run_points(K, lift, u_star, X0)
    J = res["cost_J"][res["success"]]
    return float(res["success"].mean()), float(np.median(J)) if len(J) else np.nan


def stage_scan():
    """Design-choice scan (training region x observables x gain design) on
    random VALIDATION starts, not the evaluation grid -> results/scan.csv."""
    S = kp.load_data()
    train = kp.pairs(S["train"])
    L = design_local_lqr()
    X0 = val_starts()
    rows = []

    def record(model, r, deg, n_obs, method, K, info):
        et = ctl.true_local_eigs(L["Ad"], L["Bd"], K, lift)
        a_true = ctl.rate(et)
        # always simulate (no gating on the local rate)
        succ, J = score(K, lift, L["u_star"], X0)
        rows.append(dict(model=model, r_train=r, degree=deg, n_obs=n_obs, method=method,
                         alpha_model=ctl.rate(info["eig_model"]), alpha_true=a_true,
                         gain_norm=float(np.linalg.norm(K)), success=succ, median_J=J))
        print(f"{model:8s} r={r:4.2f} d={str(deg):5s} {method:6s} | alpha model {rows[-1]['alpha_model']:7.2f} "
              f"true {a_true:9.2f} | success {100 * succ:5.1f}%  medJ {J:8.2f}", flush=True)

    lift = tl.identity_lift
    for method in METHODS:
        K, P, info = ctl.design(L["Ad"], L["Bd"], method)
        record("local", 0.0, 1, 4, method, K, info)
    for r in R_TRAIN:
        for deg in DICTS:
            A, B, lift, n_pairs = fit_region(deg, r, train)
            for method in METHODS:
                try:
                    K, P, info = ctl.design(A, B, method)
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


# ============================================================ stage: fit
def stage_fit():
    S = kp.load_data()
    train = kp.pairs(S["train"])
    H = int(round(cfg.PRED_HORIZON / cfg.DT))
    Hp = int(round(cfg.PRED_PLOT_HORIZON / cfg.DT))
    r = cfg.KOOPMAN_R_TRAIN
    val = kp.windows(S["val"], H)
    inside = kp.region_mask(val[0], r) & np.all(kp.region_mask(val[2], r), axis=1)
    val_in = kp.subset_windows(val, inside)
    test = kp.windows(S["test"], Hp, max_windows=6000)
    t_in = kp.region_mask(test[0], r) & np.all(kp.region_mask(test[2], r), axis=1)

    L = design_local_lqr()
    curves = {"taylor": (kp.multistep_rmse(L["Ad"], L["Bd"], identity_lift, kp.subset_windows(test, t_in)),
                         kp.multistep_rmse(L["Ad"], L["Bd"], identity_lift, test))}
    n_obs = {"taylor": 4}
    for name in cfg.PRED_DICTS:
        m, lift = kp.fit_dictionary(name, train, val_in, r)
        np.savez(f"{RES}/koopman_{name}.npz", **m)
        curves[name] = (kp.multistep_rmse(m["A"], m["B"], lift, kp.subset_windows(test, t_in)),
                        kp.multistep_rmse(m["A"], m["B"], lift, test))
        n_obs[name] = lift.n
        print(f"{name:6s} n={lift.n:3d} pairs={m['n_pairs']} ridge={m['ridge']:.0e} | test RMSE "
              f"@0.1s in-region {curves[name][0][19]:.4f}  all {curves[name][1][19]:.4f}")
    np.savez(f"{RES}/prediction_curves.npz",
             **{f"{k}_in": v[0] for k, v in curves.items()},
             **{f"{k}_all": v[1] for k, v in curves.items()},
             n_obs=json.dumps(n_obs), n_in=int(t_in.sum()), n_all=int(len(t_in)))
    fig_prediction()


def fig_prediction():
    d = np.load(f"{RES}/prediction_curves.npz")
    n_obs = json.loads(str(d["n_obs"]))
    names = ["taylor"] + cfg.PRED_DICTS
    t = np.arange(1, len(d["taylor_in"]) + 1) * cfg.DT
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for k, (suffix, title) in enumerate([("in", f"inside training region (|dq| <= {cfg.KOOPMAN_R_TRAIN} rad), "
                                                  f"{int(d['n_in'])} test windows"),
                                         ("all", f"all test windows ({int(d['n_all'])}), "
                                                 "incl. extrapolation")]):
        for n in names:
            ax[k].semilogy(t, d[f"{n}_{suffix}"], color=tl.COLOR[n], ls=tl.LS[n],
                           lw=1.6, label=tl.LABEL[n])
        ax[k].set_title(title, fontsize=9)
        ax[k].set_xlabel("prediction horizon [s]")
        ax[k].set_ylabel("joint-angle RMSE [rad]")
        ax[k].set_ylim(1e-4, 10)
    ax[0].legend(loc="lower right", ncol=1)
    # error vs observable count
    for hz, mk in ((0.1, "o"), (0.25, "s")):
        k = int(round(hz / cfg.DT)) - 1
        polys = [n for n in cfg.PRED_DICTS if n.startswith("poly")]
        ax[2].plot([n_obs[n] for n in polys], [d[f"{n}_in"][k] for n in polys], "-", marker=mk,
                   color=tl.TEXT2, lw=1.4, label=f"polynomial, {hz} s")
        for n in [n for n in cfg.PRED_DICTS if n.startswith("trig")]:
            ax[2].plot(n_obs[n], d[f"{n}_in"][k], marker=tl.MARKER[n], ls="none",
                       color=tl.COLOR[n], ms=9, label=f"{tl.LABEL[n]}, {hz} s")
        ax[2].axhline(d["taylor_in"][k], color=tl.COLOR["taylor"], ls="--", lw=1.2)
        ax[2].annotate(f"Taylor, {hz} s", (4, d["taylor_in"][k]), fontsize=7, color=tl.TEXT2,
                       ha="left", va="bottom")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ticks = sorted(set(n_obs[n] for n in cfg.PRED_DICTS))
    ax[2].set_xticks(ticks); ax[2].set_xticklabels([str(t) for t in ticks])
    ax[2].minorticks_off()
    ax[2].set_xlabel("number of observables")
    ax[2].set_ylabel("joint-angle RMSE [rad] (inside region)")
    ax[2].set_title("multistep error vs observable count", fontsize=9)
    ax[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    tl.title(fig, "Experiment 2: observables varied. Open-loop multistep prediction error "
                "on held-out test trajectories",
                f"same training data (|dq| <= {cfg.KOOPMAN_R_TRAIN} rad) and test inputs for "
                "every model; no controller in the loop", y=1.08)
    fig.savefig(f"{FIG}/koopman_prediction.png")
    plt.close(fig)


# ============================================================ stage: design
def stage_design():
    L = design_local_lqr()
    summary = {"local": dict(n_obs=4, suitable=True, alpha_model=ctl.rate(L["eig_cl"]),
                             alpha_true=ctl.rate(L["eig_cl"]), regularized=False,
                             stabilizable=True, detectable=True,
                             eig_true=[complex(e) for e in L["eig_cl"]])}
    for name in cfg.CONTROL_DICTS:
        m, lift = kp.load_model(name)
        row = dict(n_obs=int(lift.n), suitable=False)
        try:
            K, P, info = ctl.design(m["A"], m["B"], cfg.LQR_METHOD)
        except Exception as e:                       # e.g. Riccati failure
            K, info = None, dict(error=str(e))
        if K is not None:
            et = ctl.true_local_eigs(L["Ad"], L["Bd"], K, lift)
            row.update(alpha_model=ctl.rate(info["eig_model"]), alpha_true=ctl.rate(et),
                       regularized=bool(info.get("regularized", False)),
                       stabilizable=bool(info.get("stabilizable", True)),
                       detectable=bool(info.get("detectable", True)),
                       gain_norm=float(np.linalg.norm(K)),
                       eig_true=[complex(e) for e in et],
                       eig_model=[complex(e) for e in info["eig_model"]])
            row["suitable"] = bool(np.max(np.abs(et)) < 1.0)
            m.update(K=K, P=P)
            np.savez(f"{RES}/koopman_{name}.npz", **m)
        else:
            row.update(error=info.get("error", "not stabilizable"))
        summary[name] = row
        print(f"{name:6s} suitable={row['suitable']}  alpha model "
              f"{row.get('alpha_model', np.nan):8.2f}  true-local {row.get('alpha_true', np.nan):9.2f}")
    with open(f"{RES}/design_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda c: [c.real, c.imag])
    fig_eigs(summary)


def _s_plane(eigs):
    z = np.array([complex(*e) if isinstance(e, list) else e for e in eigs])
    return np.log(z.astype(complex)) / cfg.DT


def fig_eigs(summary=None):
    summary = summary or json.load(open(f"{RES}/design_summary.json"))
    loc = _s_plane(summary["local"]["eig_true"])
    names = cfg.CONTROL_DICTS
    fig, ax = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.6), sharey=True)
    for a, n in zip(ax, names):
        row = summary[n]
        a.axvline(0, color=tl.TEXT2, lw=0.8)
        a.plot(loc.real, loc.imag, "x", color=tl.COLOR["local"], mew=2, ms=9,
               label="local LQR (true plant)")
        if "eig_model" in row:
            sm = _s_plane(row["eig_model"])
            a.plot(sm.real, sm.imag, "o", mfc="none", color=tl.TEXT2, ms=6, mew=1,
                   label="lifted model (predicted)")
            stt = _s_plane(row["eig_true"])
            a.plot(stt.real, stt.imag, tl.MARKER[n], color=tl.COLOR[n], ms=8,
                   label="true plant, this gain")
            # pin UNSTABLE true-plant eigenvalues beyond the view to the right edge
            far = stt[stt.real > 20]
            if len(far):
                a.plot(np.full(len(far), 19.0), np.clip(far.imag, -24, 24), ">",
                       color=tl.COLOR[n], ms=10, mec=tl.TEXT)
                a.annotate(f"{len(far)} unstable,\nmax Re s = {far.real.max():+.0f}",
                           (19, 0), xytext=(-6, 10), textcoords="offset points",
                           ha="right", fontsize=7, color=tl.TEXT)
            out = int(np.sum(sm.real < -40) + np.sum(stt.real < -40))
            note = f"\n({out} fast stable eig. left of view)" if out else ""
            a.set_title(f"{tl.LABEL[n]}\nalpha true {row['alpha_true']:.2f} / model "
                        f"{row['alpha_model']:.2f} 1/s{note}", fontsize=8)
        else:
            a.set_title(f"{tl.LABEL[n]}\ndesign failed", fontsize=8)
        a.set_xlim(-40, 20); a.set_ylim(-25, 25)
        a.set_xlabel("Re s [1/s]")
    ax[0].set_ylabel("Im s [1/s]")
    ax[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    tl.title(fig, "Experiment 2: observables varied. Closed-loop eigenvalues s = ln(z)/dt "
                "(stable: Re s < 0): lifted model vs real plant, same gain",
                "Q, R, dt and the gain design (infinite-horizon LQR)", y=1.1)
    fig.savefig(f"{FIG}/koopman_eigenvalues.png")
    plt.close(fig)


# ============================================================ stage: maps
def stage_maps(which="all"):
    """Simulate every suitable Koopman controller over the conditions.
    Local LQR maps come from run_local_lqr.py (SIMULATION 1)."""
    names = koopman_names()
    print(f"=== Koopman maps: {which} conditions x {names}, "
          f"{cfg.MAP_N} x {cfg.MAP_N} initial angles each ===")
    for cond in tl.conditions(which):
        for name in names:
            res = tl.run_map(functools.partial(ctl.koopman_factory, name), cond)
            path = tl.save_map(name, cond, res)
            print(f"{cond.name:16s} {name:6s} [{res['simulator']}]  success "
                  f"{100 * res['success'].mean():5.1f}%   -> {path}", flush=True)


def load_map(name, cond=tl.MAIN):
    return tl.load_map(name, cond)


# ============================================================ stage: roa
def stage_roa():
    u_star = design_local_lqr()["u_star"]
    grid = np.linspace(-np.pi, np.pi, 201)
    out = {}
    for name in suitable_names():
        K, P, lift = ctl.controller_parts(name)
        cert = ctl.roa_certificate(K, P, lift, u_star)
        V, mask = ctl.certified_slice(P, lift, cert["rho"], grid)
        np.savez(f"{RES}/roa_{name}.npz", V=V, mask=mask, grid=grid, **cert)
        out[name] = dict(cert, area_frac=float(mask.mean()))
        print(f"roa {name:6s}: rho* = {cert['rho']:.4g} (violation {cert['rho_viol']:.4g}, "
              f"box {cert['rho_box']:.4g})  certified slice area {100 * mask.mean():.2f}%", flush=True)
    json.dump(out, open(f"{RES}/roa_summary.json", "w"), indent=1)


# ============================================================ figures
def fig_success_maps(names):
    n = len(names)
    fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True)
    cmap = ListedColormap([tl.CRITICAL, tl.GOOD])
    for a, name in zip(np.atleast_1d(ax), names):
        r = load_map(name); g = r["grid"]
        a.imshow(r["success"].astype(float), origin="lower", extent=[g[0], g[-1], g[0], g[-1]],
                 cmap=cmap, vmin=0, vmax=1, alpha=0.85)
        a.contour(g, g, r["success"].astype(float), levels=[0.5], colors=tl.TEXT, linewidths=0.8)
        a.plot(0, 0, "+", color="white", ms=10, mew=2)
        a.set_title(f"{tl.LABEL[name]}\nreaches upright from {100 * r['success'].mean():.1f}% "
                    "of starts", fontsize=8)
        a.set_xlabel(r"$\Delta q_1(0)$ [rad]"); a.grid(False)
    np.atleast_1d(ax)[0].set_ylabel(r"$\Delta q_2(0)$ [rad]")
    fig.tight_layout()
    tl.title(fig, "Experiments 1 + 2: initial angles varied (axes), controller and "
                "observables varied (panels). Green = reaches upright, red = fails",
                f"{FIXED_MAIN}; {cfg.MAP_N} x {cfg.MAP_N} starts", y=1.1)
    fig.savefig(f"{FIG}/maps_success_all.png"); plt.close(fig)


def fig_who_wins(names):
    base = load_map("local"); g = base["grid"]
    ks = [n for n in names if n != "local"]
    fig, ax = plt.subplots(1, len(ks), figsize=(3.6 * len(ks), 3.9), sharey=True)
    for a, n in zip(np.atleast_1d(ax), ks):
        r = load_map(n)
        sl, sk = base["success"], r["success"]
        cls = np.zeros(sl.shape, int)                       # 0 both fail
        cls[sl & ~sk] = 1                                   # only local
        cls[~sl & sk] = 2                                   # only Koopman
        both = sl & sk
        ratio = np.where(both, r["cost_J"] / base["cost_J"], np.nan)
        TIE = 0.01                                          # |J_K / J_local - 1| < 1 %
        cls[both & (ratio < 1 - TIE)] = 4                   # Koopman cheaper
        cls[both & (ratio > 1 + TIE)] = 3                   # local cheaper
        cls[both & (np.abs(ratio - 1) <= TIE)] = 5          # tie
        tint = {"trig1": "#f6cfe0", "poly1": "#f9e4b8", "poly2": "#c5ead9"}.get(n, "#e4e3df")
        cols = [tl.NEUTRAL, tl.COLOR["local"], tl.COLOR[n], "#b7d3f6", tint, "#ffffff"]
        a.imshow(cls, origin="lower", extent=[g[0], g[-1], g[0], g[-1]],
                 cmap=ListedColormap(cols), vmin=-0.5, vmax=5.5)
        a.contour(g, g, sl.astype(float), levels=[0.5], colors=[tl.COLOR["local"]],
                  linewidths=1.0, linestyles="--")
        labels = ["both fail", "only local converges", "only Koopman converges",
                  "both, local >1% cheaper", "both, Koopman >1% cheaper", "both, tie (within 1%)"]
        for c, l, k in zip(cols, labels, range(6)):
            a.plot([], [], "s", color=c, mec=tl.TEXT2, mew=0.5, ms=9,
                   label=f"{l} ({100 * np.mean(cls == k):.0f}%)")
        a.plot([], [], "--", color=tl.COLOR["local"], lw=1, label="local-LQR basin boundary")
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7, ncol=1)
        a.set_title(f"Local LQR vs {tl.LABEL[n]}", fontsize=8)
        a.set_xlabel(r"$\Delta q_1(0)$ [rad]"); a.grid(False)
    np.atleast_1d(ax)[0].set_ylabel(r"$\Delta q_2(0)$ [rad]")
    fig.tight_layout()
    tl.title(fig, "Who controls better from each initial angle? Reaching upright wins; "
                "if both do, lower cost J wins (within 1% = tie)",
                f"{FIXED_MAIN}; the same {cfg.MAP_N * cfg.MAP_N} starts for both", y=1.1)
    fig.savefig(f"{FIG}/maps_who_wins.png"); plt.close(fig)


def by_distance(names):
    """Main-condition maps grouped by distance of the start from upright,
    sqrt(dq1^2 + dq2^2).  Returns bands, cell counts and, per controller,
    (success % per band, median cost ratio vs local LQR per band)."""
    base = load_map("local"); g = base["grid"]
    D1, D2 = np.meshgrid(g, g); dist = np.hypot(D1, D2)
    edges = np.asarray(cfg.DISTANCE_EDGES, float)
    bands = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
             if np.any((dist >= lo) & (dist < hi))]
    masks = [(dist >= lo) & (dist < hi) for lo, hi in bands]
    out = {}
    for n in names:
        r = load_map(n)
        succ = [100 * float(r["success"][m].mean()) for m in masks]
        ratio = []
        for m in masks:
            both = m & r["success"] & base["success"]
            ratio.append(float(np.median(r["cost_J"][both] / base["cost_J"][both]))
                         if both.any() else np.nan)
        out[n] = (succ, ratio)
    return bands, [int(m.sum()) for m in masks], out


def fig_by_distance(names):
    """Is Koopman better close to the target or far from it?"""
    bands, counts, out = by_distance(names)
    x = np.arange(len(bands))
    ticks = [f"{lo:g}-{hi:g}\n({c} starts)" for (lo, hi), c in zip(bands, counts)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for n in names:
        succ, ratio = out[n]
        ax[0].plot(x, succ, color=tl.COLOR[n], ls=tl.LS[n], marker=tl.MARKER[n],
                   label=tl.LABEL[n])
        if n != "local":
            ax[1].plot(x, ratio, color=tl.COLOR[n], ls=tl.LS[n], marker=tl.MARKER[n],
                       label=tl.LABEL[n])
    ax[1].axhspan(1 - 0.01, 1 + 0.01, color=tl.GRID, zorder=0)
    ax[1].axhline(1.0, color=tl.COLOR["local"], lw=1.2, label="local LQR (= 1)")
    ax[1].annotate("below 1: Koopman cheaper; grey band: tie (within 1%)",
                   (0.02, 0.03), xycoords="axes fraction", fontsize=8, color=tl.TEXT2)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(ticks, fontsize=8)
        a.set_xlabel(r"distance of the start from upright  $\sqrt{\Delta q_1^2+\Delta q_2^2}$ (rad)")
    ax[0].set_ylabel("starts that reach upright (%)"); ax[0].set_ylim(0, 103)
    ax[0].set_title("Success rate per distance band", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower left")
    ax[1].set_ylabel("median cost J, Koopman / local LQR")
    ax[1].set_title("Cost where both reach upright", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    tl.title(fig, "Where is Koopman better? Success and cost grouped by how far the start "
                  "is from upright",
             f"{FIXED_MAIN}; same {cfg.MAP_N} x {cfg.MAP_N} starts; Koopman trained on "
             f"|dq| <= {cfg.KOOPMAN_R_TRAIN} rad per joint", y=1.1)
    fig.savefig(f"{FIG}/by_distance.png"); plt.close(fig)
    # table
    md = ["| distance from upright (rad) | starts | " + " | ".join(tl.SHORT[n] for n in names)
          + " |", "|" + "---|" * (len(names) + 2)]
    for i, ((lo, hi), c) in enumerate(zip(bands, counts)):
        cells = []
        for n in names:
            s, r = out[n][0][i], out[n][1][i]
            cells.append(f"{s:.0f}%" if n == "local" or np.isnan(r) else f"{s:.0f}% (J x{r:.3f})")
        md.append(f"| {lo:g}-{hi:g} | {c} | " + " | ".join(cells) + " |")
    md.append("\nSuccess rate; in brackets the median cost of that Koopman controller "
              "relative to local LQR where both reach upright (below 1 = Koopman cheaper).")
    open(f"{RES}/summary_distance_{TAG}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def fig_roa(names):
    """Small multiples: empirical basin (filled) + certified Lyapunov set
    (solid outline) + local-LQR certified ellipse for reference (dashed)."""
    loc = np.load(f"{RES}/roa_local.npz")
    fig, ax = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.7), sharey=True)
    for a, n in zip(np.atleast_1d(ax), names):
        r = load_map(n); g = r["grid"]
        roa = np.load(f"{RES}/roa_{n}.npz"); gg = roa["grid"]
        a.contourf(g, g, r["success"].astype(float), levels=[0.5, 1.5],
                   colors=[tl.COLOR[n]], alpha=0.22)
        a.contour(g, g, r["success"].astype(float), levels=[0.5], colors=[tl.COLOR[n]],
                  linewidths=1.0, linestyles=":")
        if roa["mask"].any():
            a.contourf(gg, gg, roa["mask"].astype(float), levels=[0.5, 1.5],
                       colors=[tl.COLOR[n]], alpha=0.65)
            a.contour(gg, gg, roa["mask"].astype(float), levels=[0.5], colors=[tl.TEXT],
                      linewidths=1.2)
        if n != "local" and loc["mask"].any():
            a.contour(gg, gg, loc["mask"].astype(float), levels=[0.5],
                      colors=[tl.COLOR["local"]], linewidths=1.4, linestyles="--")
        a.plot(0, 0, "+", color=tl.TEXT, ms=9, mew=1.5)
        a.set_title(f"{tl.LABEL[n]}\ncertified {100 * roa['mask'].mean():.1f}%  |  "
                    f"empirical {100 * r['success'].mean():.0f}%", fontsize=8)
        a.set_xlabel(r"$\Delta q_1$ [rad]"); a.set_aspect("equal")
    np.atleast_1d(ax)[0].set_ylabel(r"$\Delta q_2$ [rad]")
    fig.tight_layout()
    tl.title(fig, "Regions of attraction by controller. Light fill: starts that reach upright "
                "(simulated). Dark fill: sampled Lyapunov certificate {V <= rho*}. "
                "Dashed blue: local-LQR certificate",
                f"{FIXED_MAIN}", y=1.12)
    fig.savefig(f"{FIG}/roa_shapes.png"); plt.close(fig)

    # zoom: certified shapes only, overlaid (contour of V at rho*: smooth)
    fig, a = plt.subplots(figsize=(5.4, 4.8))
    span = 0.0
    for n in names:
        roa = np.load(f"{RES}/roa_{n}.npz"); gg = roa["grid"]
        if roa["mask"].any():
            a.contour(gg, gg, roa["V"], levels=[float(roa["rho"])], colors=[tl.COLOR[n]],
                      linestyles=[tl.LS[n]], linewidths=2)
            a.plot([], [], color=tl.COLOR[n], ls=tl.LS[n],
                   label=f"{tl.LABEL[n]}  ({100 * roa['mask'].mean():.2f}% of slice)")
            D1, D2 = np.meshgrid(gg, gg)
            span = max(span, np.abs(D1[roa["mask"]]).max(), np.abs(D2[roa["mask"]]).max())
    span = 1.2 * span if span > 0 else 1.0
    a.set_xlim(-span, span); a.set_ylim(-span, span); a.set_aspect("equal")
    a.set_xlabel(r"$\Delta q_1$ [rad]"); a.set_ylabel(r"$\Delta q_2$ [rad]")
    a.legend(loc="upper left", fontsize=7)
    a.set_title("Certified sets {V <= rho*}, V = psi' P psi from each controller's Riccati.\n"
                "Torque saturation, not the model, limits them: local, state-only and\n"
                "trig coincide (sin x ~ x at this size); quadratic is smaller and bent.\n"
                f"Fixed: {FIXED_MAIN}", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/roa_certified_overlay.png"); plt.close(fig)


def _run_one(name, x0):
    """One DFKI run (for time-series figures)."""
    fac = ctl.local_lqr_factory if name == "local" else functools.partial(ctl.koopman_factory, name)
    ctrl, u_star = fac()
    return tl.simulate(ctrl, x0)


def pick_showcase(names):
    """Initial angles where local fails and the best Koopman succeeds
    (closest such cell to the target), plus one where both succeed."""
    base = load_map("local"); g = base["grid"]
    best = max([n for n in names if n != "local"], key=lambda n: load_map(n)["success"].mean())
    rk = load_map(best)
    D1, D2 = np.meshgrid(g, g); dist = np.hypot(D1, D2)
    only = rk["success"] & ~base["success"]
    both = rk["success"] & base["success"]
    pick = {}
    if only.any():
        i, j = np.unravel_index(np.argmin(np.where(only, dist, np.inf)), dist.shape)
        pick["koopman_only"] = (g[j], g[i])
    target = np.where(both, np.abs(dist - 1.4), np.inf)
    i, j = np.unravel_index(np.argmin(target), dist.shape)
    pick["both"] = (g[j], g[i])
    return best, pick


def fig_timeseries(names):
    best, pick = pick_showcase(names)
    for key, (d1, d2) in pick.items():
        x0 = cfg.X_STAR + np.array([d1, d2, 0, 0])
        fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        for n in ("local", best):
            T, X, U = _run_one(n, x0)
            k = min(len(X), len(U)); T, X, U = T[:k], X[:k], U[:k]
            dX = X - cfg.X_STAR
            dX[:, :2] = np.pi - np.mod(np.pi - dX[:, :2], 2 * np.pi)
            for j in range(2):
                ax[j].plot(T, dX[:, j], color=tl.COLOR[n], ls=tl.LS[n], lw=1.6,
                           label=tl.LABEL[n])
                ax[2 + j].plot(T, U[:, j], color=tl.COLOR[n], ls=tl.LS[n], lw=1.4)
        for j in range(2):
            ax[j].set_ylabel(f"$\\Delta q_{j + 1}$ [rad]")
            ax[2 + j].set_ylabel(f"$u_{j + 1}$ [N m]")
            for s in (1, -1):
                ax[2 + j].axhline(s * cfg.TORQUE_LIMIT[j], color=tl.TEXT2, ls=":", lw=0.8)
        ax[0].legend(loc="upper right")
        ax[3].set_xlabel("time [s]")
        fig.tight_layout()
        verdict = ("only Koopman reaches upright" if key == "koopman_only"
                   else "both reach upright")
        tl.title(fig, f"Joint angles and torques vs time from initial angles dq0 = "
                    f"[{d1:.2f}, {d2:.2f}] rad: Local LQR vs {tl.LABEL[best]} ({verdict})",
                    f"{FIXED_MAIN}; DFKI simulator", y=1.04)
        fig.savefig(f"{FIG}/timeseries_{key}.png"); plt.close(fig)
    return best, pick


def fig_convergence(names, pick):
    """(a) e(t) = sqrt(dx'Q dx) on a log scale with the theoretical local
    slope; (b) theoretical local rate vs measured tail rates;
    (c) transient constant c_hat."""
    summary = json.load(open(f"{RES}/design_summary.json"))
    d1, d2 = pick["both"]
    x0 = cfg.X_STAR + np.array([d1, d2, 0, 0])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for n in names:
        T, X, U = _run_one(n, x0)
        dX = X - cfg.X_STAR
        dX[:, :2] = np.pi - np.mod(np.pi - dX[:, :2], 2 * np.pi)
        e = np.sqrt(np.einsum("ki,ij,kj->k", dX, cfg.Q, dX))
        ax[0].semilogy(T, e, color=tl.COLOR[n], ls=tl.LS[n], lw=1.6, label=tl.LABEL[n])
        a = summary[n]["alpha_true"]
        k0 = np.argmax(e < cfg.RATE_E_HI)
        tt = np.linspace(T[k0], T[-1], 50)
        ax[0].semilogy(tt, e[k0] * np.exp(-a * (tt - T[k0])), color=tl.COLOR[n], lw=0.8, ls=":")
    ax[0].set_ylim(1e-7, 20)
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel(r"$\sqrt{\delta x^T Q \delta x}$")
    ax[0].set_title(f"decay from dq0 = [{d1:.2f}, {d2:.2f}]; dotted = local theory "
                    r"$e^{-\alpha t}$", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower left")
    for i, n in enumerate(names):
        r = load_map(n)
        a_hat = r["alpha_hat"][r["success"]]
        c_hat = r["c_hat"][r["success"]]
        q = np.nanpercentile(a_hat, [25, 50, 75])
        ax[1].errorbar(i, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt=tl.MARKER[n],
                       color=tl.COLOR[n], ms=9, capsize=4, lw=1.5)
        ax[1].plot(i, summary[n]["alpha_true"], "_", color=tl.TEXT, ms=22, mew=2)
        qc = np.nanpercentile(c_hat, [25, 50, 75])
        ax[2].errorbar(i, qc[1], yerr=[[qc[1] - qc[0]], [qc[2] - qc[1]]], fmt=tl.MARKER[n],
                       color=tl.COLOR[n], ms=9, capsize=4, lw=1.5)
    for a in ax[1:]:
        a.set_xticks(range(len(names)))
        a.set_xticklabels([tl.SHORT[n] for n in names], rotation=25, ha="right", fontsize=7)
    ax[1].plot([], [], "_", color=tl.TEXT, ms=14, mew=2, label="local theory (true plant)")
    ax[1].plot([], [], "o", color=tl.TEXT2, label="measured tail rate: median, IQR")
    ax[1].legend(fontsize=7, loc="upper left")
    ax[1].set_ylabel(r"exponential rate $\alpha$ [1/s]")
    ax[1].set_title("local exponential rate: theory vs measured", fontsize=9)
    ax[2].set_ylabel(r"transient constant $\hat c$  (median, IQR)")
    ax[2].set_yscale("log")
    ax[2].set_title(r"overshoot / saturation: $e(t) \leq \hat c\, e(0)\, e^{-\hat\alpha t}$",
                    fontsize=9)
    fig.tight_layout()
    tl.title(fig, "How fast each controller converges: near the target all share the "
                "linear rate; differences come from the transient (saturation, overshoot)",
                f"{FIXED_MAIN}; rates measured over the starts each controller stabilizes", y=1.1)
    fig.savefig(f"{FIG}/convergence_rates.png"); plt.close(fig)


def fig_scan():
    """Sensitivity: validation success vs training radius x dictionary."""
    rows = list(csv.DictReader(open(f"{RES}/scan.csv")))
    dicts = ["1", "2", "3", "trig1", "trig2"]
    names = ["poly1", "poly2", "poly3", "trig1", "trig2"]
    radii = sorted({float(r["r_train"]) for r in rows if r["model"] == "koopman"})
    loc = {r["method"]: float(r["success"]) for r in rows if r["model"] == "local"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for a, meth in zip(ax, ("inf", "fh0.5")):
        M = np.full((len(dicts), len(radii)), np.nan)
        for r in rows:
            if r["model"] == "koopman" and r["method"] == meth:
                M[dicts.index(r["degree"]), radii.index(float(r["r_train"]))] = 100 * float(r["success"])
        im = a.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                       color="white" if v > 60 else tl.TEXT)
        a.set_xticks(range(len(radii))); a.set_xticklabels([f"{r:.2f}" for r in radii])
        a.set_yticks(range(len(dicts))); a.set_yticklabels([tl.LABEL[n] for n in names], fontsize=8)
        a.set_xlabel("training region  r_train [rad]")
        a.set_title(f"{'infinite-horizon LQR' if meth == 'inf' else 'receding-horizon gain, 0.5 s'}"
                    f"  (local LQR: {100 * loc[meth]:.0f}%)", fontsize=9)
        a.grid(False)
        fig.colorbar(im, ax=a, label="validation success [%]")
    fig.tight_layout()
    tl.title(fig, "Design choice (not a result): validation success vs training region "
                "and observables, for two gain designs",
                f"400 random validation starts (not the reported grid); {FIXED_MAIN}", y=1.1)
    fig.savefig(f"{FIG}/scan_sensitivity.png"); plt.close(fig)


def summary_table(names):
    design = json.load(open(f"{RES}/design_summary.json"))
    roa_path = f"{RES}/roa_summary.json"              # from --stage extra (optional)
    roa = json.load(open(roa_path)) if os.path.exists(roa_path) else {}
    base = load_map("local")
    rows = []
    for n in ["local"] + cfg.CONTROL_DICTS:
        d = design[n]
        row = dict(controller=tl.LABEL[n], n_obs=d["n_obs"], suitable=d["suitable"],
                   alpha_model=d.get("alpha_model", np.nan), alpha_true_local=d.get("alpha_true", np.nan))
        if n in names:
            r = load_map(n); s = r["success"]
            both = s & base["success"]
            row.update(
                success_pct=100 * s.mean(),
                certified_pct=100 * roa[n]["area_frac"] if n in roa else np.nan,
                only_this_pct=100 * np.mean(s & ~base["success"]) if n != "local" else 0.0,
                only_local_pct=100 * np.mean(~s & base["success"]) if n != "local" else 0.0,
                median_J=float(np.median(r["cost_J"][s])),
                median_J_ratio_vs_local=float(np.median(r["cost_J"][both] / base["cost_J"][both])),
                median_settling_s=float(np.nanmedian(r["settling_time"][s])),
                median_alpha_hat=float(np.nanmedian(r["alpha_hat"][s])),
                median_c_hat=float(np.nanmedian(r["c_hat"][s])),
                median_rmse_q=float(np.median(r["rmse_q"][s])),
                median_effort=float(np.median(r["effort"][s])),
                mean_sat_pct=100 * float(np.mean(r["sat_frac"][s])))
        rows.append(row)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r: keys.append(k) if k not in keys else None
    with open(f"{RES}/summary_{TAG}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    fmt = lambda v: (f"{v:.2f}" if isinstance(v, float) else str(v))
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in rows]
    open(f"{RES}/summary_{TAG}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def _robust_rates(names):
    """Success (%) of every controller under every condition.  Returns the
    conditions that have maps for all controllers, and their rates."""
    conds, rates, missing = [], {}, []
    for cond in tl.conditions("all"):
        try:
            rates[cond.name] = {n: 100 * load_map(n, cond)["success"].mean() for n in names}
            conds.append(cond)
        except FileNotFoundError as e:
            missing.append(str(e).split(";")[0])
    return conds, rates, missing


def fig_robustness(names):
    """Robustness: one variable changed at a time, the rest at main.
    One panel per robustness variable (config.py, section 1)."""
    conds, rates, missing = _robust_rates(names)
    if missing:
        print("robustness figure skipped, missing maps:\n  " + "\n  ".join(missing))
        return None
    titles = {"torque": ("torque limit [joint 1, joint 2] (N\u00b7m)", "Torque limit varied"),
              "velocity": ("initial velocity (qd1, qd2) (rad/s)", "Initial velocity varied"),
              "model": ("design-model masses (\u00d7 true)",
                        "Model error varied (physics designs only)"),
              "plant": ("arm masses (\u00d7 nominal)",
                        "Plant changed (all controllers blind)"),
              "noise": ("sensor noise level (\u00d7 NOISE_BASE)", "Sensor noise varied")}
    main_tick = {"torque": f"[{', '.join(f'{v:g}' for v in tl.MAIN.torque_limit)}]",
                 "velocity": f"({', '.join(f'{v:g}' for v in tl.MAIN.v0)})",
                 "model": f"\u00d7{tl.MAIN.model_error:g}",
                 "plant": f"\u00d7{tl.MAIN.plant_mass:g}",
                 "noise": f"\u00d7{tl.MAIN.noise_level:g}"}
    groups = [g for g in tl.GROUPS if any(c.group == g for c in conds)]
    fig, ax = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.2), squeeze=False)
    for a, g in zip(ax[0], groups):
        members = [c for c in conds if c.group == g]
        if g != "velocity":
            members.sort(key=lambda c: c.value)
        # put the main condition at its place in the sequence
        main_val = {"torque": 1.0, "model": tl.MAIN.model_error, "plant": tl.MAIN.plant_mass,
                    "noise": tl.MAIN.noise_level}.get(g)
        seq = ([tl.MAIN] + members if g == "velocity" else
               sorted([tl.MAIN] + members,
                      key=lambda c: main_val if c.group == "main" else c.value))
        ticks = [main_tick[g] + " main" if c.group == "main" else
                 {"torque": lambda c: f"[{c.torque_limit[0]:g}, {c.torque_limit[1]:g}]",
                  "velocity": lambda c: f"({c.v0[0]:g}, {c.v0[1]:g})",
                  "model": lambda c: f"\u00d7{c.model_error:g}",
                  "plant": lambda c: f"\u00d7{c.plant_mass:g}",
                  "noise": lambda c: f"\u00d7{c.noise_level:g}"}[g](c) for c in seq]
        k = len(names)
        for i, n in enumerate(names):
            x = np.arange(len(seq)) + ((i - (k - 1) / 2) * 0.1 if g == "velocity" else 0)
            a.plot(x, [rates[c.name][n] for c in seq], color=tl.COLOR[n],
                   ls="none" if g == "velocity" else tl.LS[n], marker=tl.MARKER[n],
                   label=tl.LABEL[n])
        a.set_xticks(range(len(seq))); a.set_xticklabels(ticks, fontsize=8)
        a.set_xlabel(titles[g][0]); a.set_title(titles[g][1], fontsize=9)
        a.set_ylim(0, 103)
    ax[0, 0].set_ylabel("starts that reach upright (%)")
    ax[0, 0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    tl.title(fig, "Robustness: one variable changed at a time; success over the same "
                f"{cfg.MAP_N} x {cfg.MAP_N} initial angles",
                "everything else at the main condition: " + tl.MAIN.describe(), y=1.1)
    fig.savefig(f"{FIG}/robustness.png"); plt.close(fig)
    # table: condition x controller
    head = "| condition (one variable changed) | " + " | ".join(tl.SHORT[n] for n in names) + " |"
    md = [head, "|" + "---|" * (len(names) + 1)]
    for c in conds:
        md.append(f"| {c.label()} | " + " | ".join(f"{rates[c.name][n]:.1f}%" for n in names) + " |")
    open(f"{RES}/summary_robustness_{TAG}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return rates


def slide_table(names):
    """Five-column summary for a slide (main condition)."""
    design = json.load(open(f"{RES}/design_summary.json"))
    base = load_map("local")
    md = ["| controller | observables N | reaches upright | local rate alpha (1/s) | "
          "median cost vs local LQR |", "|---|---|---|---|---|"]
    for n in ["local"] + cfg.OBSERVABLES:
        d = design[n]
        if n in names:
            r = load_map(n); both = r["success"] & base["success"]
            ratio = np.median(r["cost_J"][both] / base["cost_J"][both])
            md.append(f"| {tl.LABEL[n]} | {d['n_obs']} | {100 * r['success'].mean():.1f}% | "
                      f"{d['alpha_true']:.2f} | {ratio:.3f} |")
        else:
            md.append(f"| {tl.LABEL[n]} | {d['n_obs']} | not usable: gain destabilizes "
                      "the real plant | \u2014 | \u2014 |")
    md.append(f"\nFixed: {FIXED_MAIN}; {cfg.MAP_N} x {cfg.MAP_N} initial angles.")
    open(f"{RES}/summary_slide_{TAG}.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def stage_figures():
    """Core comparison figures and tables."""
    names = suitable_names()
    fig_success_maps(names)
    fig_who_wins(names)
    fig_by_distance(names)
    fig_timeseries(names)
    summary_table(names)
    slide_table(names)
    fig_robustness(names)


def stage_extra():
    """Optional analysis, not needed for the comparison: the design scan that
    chose KOOPMAN_R_TRAIN and the gain design, sampled Lyapunov certificates,
    and convergence-rate figures."""
    stage_scan()
    fig_scan()
    stage_roa()
    names = suitable_names()
    fig_roa(names)
    _, pick = pick_showcase(names)
    fig_convergence(names, pick)
    summary_table(names)                    # now with certified areas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "data", "fit", "design", "maps", "figures", "extra",
                             "scan", "roa"],
                    help="all = data, fit, design, maps, figures; extra = scan, roa and "
                         "their figures (optional)")
    ap.add_argument("--maps", default="all", choices=["main", "robust", "all"],
                    help="which conditions the maps stage simulates (config.py, section 1)")
    a = ap.parse_args()
    if a.stage in ("all", "data"): stage_data()
    if a.stage in ("all", "fit"): stage_fit()
    if a.stage in ("all", "design"): stage_design()
    if a.stage in ("all", "maps"): stage_maps(a.maps)
    if a.stage in ("all", "figures"): stage_figures()
    if a.stage == "extra": stage_extra()
    if a.stage == "scan": stage_scan()
    if a.stage == "roa": stage_roa()


if __name__ == "__main__":
    main()
