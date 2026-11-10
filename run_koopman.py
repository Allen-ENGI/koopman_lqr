"""Stages 4-5: Koopman predictors, LQR on each lifted model, and the
stability / convergence / region-of-attraction comparison with local LQR.

    python collect_data.py                 # stage 3 (once)
    python scan_designs.py                 # design-choice scan (validation starts)
    python run_koopman.py                  # all stages below
    python run_koopman.py --stage fit      # or: design | maps | roa | figures
    python run_koopman.py --fast           # maps with the vectorized sweep
                                           # (identical plant, ideal only)
Fairness rules: same plant, Q, R, torque limit, dt, initial states and
training data for every model; same gain design (cfg.LQR_METHOD) for all.
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
import collect_data as cd
import koopman as kp
import stability as st
import style
from controllers import LiftedLQRController
from fast_sweep import identity_lift
from run_local_lqr import design_local_lqr

TAG = f"{cfg.PLANT_PROFILE}-{cfg.REALISM}"
RES, FIG = "results", "figures"
MAP_N = 31
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
style.apply()


# ============================================================ controllers
def local_factory():
    d = design_local_lqr()
    return LiftedLQRController(d["K"], cfg.X_STAR, d["u_star"], cfg.TORQUE_LIMIT,
                               name="local"), d["u_star"]


def koopman_factory(name):
    m, lift = kp.load_model(name)
    u_star = design_local_lqr()["u_star"]
    return LiftedLQRController(m["K"], cfg.X_STAR, u_star, cfg.TORQUE_LIMIT,
                               lift=lift, name=name), u_star


def controller_parts(name):
    """(K, P, lift) for any controller name."""
    if name == "local":
        d = design_local_lqr()
        return d["K"], d["P"], identity_lift
    m, lift = kp.load_model(name)
    return m["K"], m["P"], lift


def suitable_names():
    info = json.load(open(f"{RES}/design_summary.json"))
    return ["local"] + [n for n in cfg.CONTROL_DICTS if info[n]["suitable"]]


# ============================================================ stage: fit
def stage_fit():
    S = cd.load()
    train = cd.pairs(S["train"])
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
            ax[k].semilogy(t, d[f"{n}_{suffix}"], color=style.COLOR[n], ls=style.LS[n],
                           lw=1.6, label=style.LABEL[n])
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
                   color=style.TEXT2, lw=1.4, label=f"polynomial, {hz} s")
        for n in [n for n in cfg.PRED_DICTS if n.startswith("trig")]:
            ax[2].plot(n_obs[n], d[f"{n}_in"][k], marker=style.MARKER[n], ls="none",
                       color=style.COLOR[n], ms=9, label=f"{style.LABEL[n]}, {hz} s")
        ax[2].axhline(d["taylor_in"][k], color=style.COLOR["taylor"], ls="--", lw=1.2)
        ax[2].annotate(f"Taylor, {hz} s", (4, d["taylor_in"][k]), fontsize=7, color=style.TEXT2,
                       ha="left", va="bottom")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ticks = sorted(set(n_obs[n] for n in cfg.PRED_DICTS))
    ax[2].set_xticks(ticks); ax[2].set_xticklabels([str(t) for t in ticks])
    ax[2].minorticks_off()
    ax[2].set_xlabel("number of observables")
    ax[2].set_ylabel("joint-angle RMSE [rad] (inside region)")
    ax[2].set_title("multistep error vs observable count", fontsize=9)
    ax[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Open-loop multistep prediction on held-out test trajectories", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/koopman_prediction.png")
    plt.close(fig)


# ============================================================ stage: design
def stage_design():
    L = design_local_lqr()
    summary = {"local": dict(n_obs=4, suitable=True, alpha_model=st.rate(L["eig_cl"]),
                             alpha_true=st.rate(L["eig_cl"]), regularized=False,
                             stabilizable=True, detectable=True,
                             eig_true=[complex(e) for e in L["eig_cl"]])}
    for name in cfg.CONTROL_DICTS:
        m, lift = kp.load_model(name)
        row = dict(n_obs=int(lift.n), suitable=False)
        try:
            K, P, info = st.design(m["A"], m["B"], cfg.LQR_METHOD)
        except Exception as e:                       # e.g. Riccati failure
            K, info = None, dict(error=str(e))
        if K is not None:
            et = st.true_local_eigs(L["Ad"], L["Bd"], K, lift)
            row.update(alpha_model=st.rate(info["eig_model"]), alpha_true=st.rate(et),
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
        a.axvline(0, color=style.TEXT2, lw=0.8)
        a.plot(loc.real, loc.imag, "x", color=style.COLOR["local"], mew=2, ms=9,
               label="local LQR (true plant)")
        if "eig_model" in row:
            sm = _s_plane(row["eig_model"])
            a.plot(sm.real, sm.imag, "o", mfc="none", color=style.TEXT2, ms=6, mew=1,
                   label="lifted model (predicted)")
            stt = _s_plane(row["eig_true"])
            a.plot(stt.real, stt.imag, style.MARKER[n], color=style.COLOR[n], ms=8,
                   label="true plant, this gain")
            # pin UNSTABLE true-plant eigenvalues beyond the view to the right edge
            far = stt[stt.real > 20]
            if len(far):
                a.plot(np.full(len(far), 19.0), np.clip(far.imag, -24, 24), ">",
                       color=style.COLOR[n], ms=10, mec=style.TEXT)
                a.annotate(f"{len(far)} unstable,\nmax Re s = {far.real.max():+.0f}",
                           (19, 0), xytext=(-6, 10), textcoords="offset points",
                           ha="right", fontsize=7, color=style.TEXT)
            out = int(np.sum(sm.real < -40) + np.sum(stt.real < -40))
            note = f"\n({out} fast stable eig. left of view)" if out else ""
            a.set_title(f"{style.LABEL[n]}\nalpha true {row['alpha_true']:.2f} / model "
                        f"{row['alpha_model']:.2f} 1/s{note}", fontsize=8)
        else:
            a.set_title(f"{style.LABEL[n]}\ndesign failed", fontsize=8)
        a.set_xlim(-40, 20); a.set_ylim(-25, 25)
        a.set_xlabel("Re s [1/s]")
    ax[0].set_ylabel("Im s [1/s]")
    ax[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Closed-loop eigenvalues, s = ln(z)/dt  (stable: Re s < 0). "
                 "The guarantee holds for the model; the true plant decides.", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/koopman_eigenvalues.png")
    plt.close(fig)


# ============================================================ stage: maps
def stage_maps(fast=False):
    names = suitable_names()
    for name in names:
        path = f"{RES}/map_{name}_{TAG}.npz"
        if fast:
            import fast_sweep as fs
            K, P, lift = controller_parts(name)
            res = fs.run_map(K, lift, design_local_lqr()["u_star"], n=MAP_N)
        else:
            from sweep import run_map
            fac = local_factory if name == "local" else functools.partial(koopman_factory, name)
            res = run_map(fac, n=MAP_N)
        np.savez(path, **res)
        print(f"map {name:6s}: success {100 * res['success'].mean():5.1f}%", flush=True)


def load_map(name):
    return dict(np.load(f"{RES}/map_{name}_{TAG}.npz"))


# ============================================================ stage: roa
def stage_roa():
    u_star = design_local_lqr()["u_star"]
    grid = np.linspace(-np.pi, np.pi, 201)
    out = {}
    for name in suitable_names():
        K, P, lift = controller_parts(name)
        cert = st.roa_certificate(K, P, lift, u_star)
        V, mask = st.certified_slice(P, lift, cert["rho"], grid)
        np.savez(f"{RES}/roa_{name}.npz", V=V, mask=mask, grid=grid, **cert)
        out[name] = dict(cert, area_frac=float(mask.mean()))
        print(f"roa {name:6s}: rho* = {cert['rho']:.4g} (violation {cert['rho_viol']:.4g}, "
              f"box {cert['rho_box']:.4g})  certified slice area {100 * mask.mean():.2f}%", flush=True)
    json.dump(out, open(f"{RES}/roa_summary.json", "w"), indent=1)


# ============================================================ figures
def fig_success_maps(names):
    n = len(names)
    fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True)
    cmap = ListedColormap([style.CRITICAL, style.GOOD])
    for a, name in zip(np.atleast_1d(ax), names):
        r = load_map(name); g = r["grid"]
        a.imshow(r["success"].astype(float), origin="lower", extent=[g[0], g[-1], g[0], g[-1]],
                 cmap=cmap, vmin=0, vmax=1, alpha=0.85)
        a.contour(g, g, r["success"].astype(float), levels=[0.5], colors=style.TEXT, linewidths=0.8)
        a.plot(0, 0, "+", color="white", ms=10, mew=2)
        a.set_title(f"{style.LABEL[name]}\nconverges from {100 * r['success'].mean():.0f}% of slice",
                    fontsize=8)
        a.set_xlabel(r"$\Delta q_1(0)$ [rad]"); a.grid(False)
    np.atleast_1d(ax)[0].set_ylabel(r"$\Delta q_2(0)$ [rad]")
    fig.suptitle("Experiment 1: where each controller converges (green) or fails (red), "
                 "velocities start at 0  [DFKI simulator]", fontsize=10)
    fig.tight_layout()
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
        cols = [style.NEUTRAL, style.COLOR["local"], style.COLOR[n], "#b7d3f6",
                "#f6cfe0" if n == "trig1" else "#f2e4cf", "#ffffff"]
        a.imshow(cls, origin="lower", extent=[g[0], g[-1], g[0], g[-1]],
                 cmap=ListedColormap(cols), vmin=-0.5, vmax=5.5)
        a.contour(g, g, sl.astype(float), levels=[0.5], colors=[style.COLOR["local"]],
                  linewidths=1.0, linestyles="--")
        labels = ["both fail", "only local converges", "only Koopman converges",
                  "both, local >1% cheaper", "both, Koopman >1% cheaper", "both, tie (within 1%)"]
        for c, l, k in zip(cols, labels, range(6)):
            a.plot([], [], "s", color=c, mec=style.TEXT2, mew=0.5, ms=9,
                   label=f"{l} ({100 * np.mean(cls == k):.0f}%)")
        a.plot([], [], "--", color=style.COLOR["local"], lw=1, label="local-LQR basin boundary")
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7, ncol=1)
        a.set_title(f"local LQR vs {style.LABEL[n]}", fontsize=8)
        a.set_xlabel(r"$\Delta q_1(0)$ [rad]"); a.grid(False)
    np.atleast_1d(ax)[0].set_ylabel(r"$\Delta q_2(0)$ [rad]")
    fig.suptitle("Who controls better? Success first, then the common cost J", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/maps_who_wins.png"); plt.close(fig)


def fig_roa(names):
    """Small multiples: empirical basin (filled) + certified Lyapunov set
    (solid outline) + local-LQR certified ellipse for reference (dashed)."""
    loc = np.load(f"{RES}/roa_local.npz")
    fig, ax = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.7), sharey=True)
    for a, n in zip(np.atleast_1d(ax), names):
        r = load_map(n); g = r["grid"]
        roa = np.load(f"{RES}/roa_{n}.npz"); gg = roa["grid"]
        a.contourf(g, g, r["success"].astype(float), levels=[0.5, 1.5],
                   colors=[style.COLOR[n]], alpha=0.22)
        a.contour(g, g, r["success"].astype(float), levels=[0.5], colors=[style.COLOR[n]],
                  linewidths=1.0, linestyles=":")
        if roa["mask"].any():
            a.contourf(gg, gg, roa["mask"].astype(float), levels=[0.5, 1.5],
                       colors=[style.COLOR[n]], alpha=0.65)
            a.contour(gg, gg, roa["mask"].astype(float), levels=[0.5], colors=[style.TEXT],
                      linewidths=1.2)
        if n != "local" and loc["mask"].any():
            a.contour(gg, gg, loc["mask"].astype(float), levels=[0.5],
                      colors=[style.COLOR["local"]], linewidths=1.4, linestyles="--")
        a.plot(0, 0, "+", color=style.TEXT, ms=9, mew=1.5)
        a.set_title(f"{style.LABEL[n]}\ncertified {100 * roa['mask'].mean():.1f}%  |  "
                    f"empirical {100 * r['success'].mean():.0f}%", fontsize=8)
        a.set_xlabel(r"$\Delta q_1$ [rad]"); a.set_aspect("equal")
    np.atleast_1d(ax)[0].set_ylabel(r"$\Delta q_2$ [rad]")
    fig.tight_layout()
    fig.suptitle("Regions of attraction on the zero-velocity slice.  Light fill: empirical basin "
                 "(simulated).  Dark fill: sampled Lyapunov certificate {V <= rho*}.  "
                 "Dashed blue: local-LQR certified set.", fontsize=9, y=1.05)
    fig.savefig(f"{FIG}/roa_shapes.png"); plt.close(fig)

    # zoom: certified shapes only, overlaid (contour of V at rho*: smooth)
    fig, a = plt.subplots(figsize=(5.4, 4.8))
    span = 0.0
    for n in names:
        roa = np.load(f"{RES}/roa_{n}.npz"); gg = roa["grid"]
        if roa["mask"].any():
            a.contour(gg, gg, roa["V"], levels=[float(roa["rho"])], colors=[style.COLOR[n]],
                      linestyles=[style.LS[n]], linewidths=2)
            a.plot([], [], color=style.COLOR[n], ls=style.LS[n],
                   label=f"{style.LABEL[n]}  ({100 * roa['mask'].mean():.2f}% of slice)")
            D1, D2 = np.meshgrid(gg, gg)
            span = max(span, np.abs(D1[roa["mask"]]).max(), np.abs(D2[roa["mask"]]).max())
    span = 1.2 * span if span > 0 else 1.0
    a.set_xlim(-span, span); a.set_ylim(-span, span); a.set_aspect("equal")
    a.set_xlabel(r"$\Delta q_1$ [rad]"); a.set_ylabel(r"$\Delta q_2$ [rad]")
    a.legend(loc="upper left", fontsize=7)
    a.set_title("Certified sets {V <= rho*}, V = psi' P psi from each controller's Riccati.\n"
                "Torque saturation, not the model, limits them: local, state-only and\n"
                "trig coincide (sin x ~ x at this size); quadratic is smaller and bent.",
                fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/roa_certified_overlay.png"); plt.close(fig)


def _run_one(name, x0):
    """One DFKI run (for time-series figures)."""
    from run_local_lqr import simulate
    from plant_utils import make_plant
    fac = local_factory if name == "local" else functools.partial(koopman_factory, name)
    ctrl, u_star = fac()
    return simulate(make_plant(), ctrl, x0)


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
                ax[j].plot(T, dX[:, j], color=style.COLOR[n], ls=style.LS[n], lw=1.6,
                           label=style.LABEL[n])
                ax[2 + j].plot(T, U[:, j], color=style.COLOR[n], ls=style.LS[n], lw=1.4)
        for j in range(2):
            ax[j].set_ylabel(f"$\\Delta q_{j + 1}$ [rad]")
            ax[2 + j].set_ylabel(f"$u_{j + 1}$ [N m]")
            for s in (1, -1):
                ax[2 + j].axhline(s * cfg.TORQUE_LIMIT[j], color=style.TEXT2, ls=":", lw=0.8)
        ax[0].legend(loc="upper right")
        ax[3].set_xlabel("time [s]")
        fig.suptitle(f"Joint angles and torques from dq0 = [{d1:.2f}, {d2:.2f}] rad "
                     f"({'only Koopman converges' if key == 'koopman_only' else 'both converge'})"
                     "  [DFKI simulator]", fontsize=10)
        fig.tight_layout()
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
        ax[0].semilogy(T, e, color=style.COLOR[n], ls=style.LS[n], lw=1.6, label=style.LABEL[n])
        a = summary[n]["alpha_true"]
        k0 = np.argmax(e < cfg.RATE_E_HI)
        tt = np.linspace(T[k0], T[-1], 50)
        ax[0].semilogy(tt, e[k0] * np.exp(-a * (tt - T[k0])), color=style.COLOR[n], lw=0.8, ls=":")
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
        ax[1].errorbar(i, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt=style.MARKER[n],
                       color=style.COLOR[n], ms=9, capsize=4, lw=1.5)
        ax[1].plot(i, summary[n]["alpha_true"], "_", color=style.TEXT, ms=22, mew=2)
        qc = np.nanpercentile(c_hat, [25, 50, 75])
        ax[2].errorbar(i, qc[1], yerr=[[qc[1] - qc[0]], [qc[2] - qc[1]]], fmt=style.MARKER[n],
                       color=style.COLOR[n], ms=9, capsize=4, lw=1.5)
    for a in ax[1:]:
        a.set_xticks(range(len(names)))
        a.set_xticklabels([style.LABEL[n].replace("Koopman: ", "K: ") for n in names],
                          rotation=25, ha="right", fontsize=7)
    ax[1].plot([], [], "_", color=style.TEXT, ms=14, mew=2, label="local theory (true plant)")
    ax[1].plot([], [], "o", color=style.TEXT2, label="measured tail rate: median, IQR")
    ax[1].legend(fontsize=7, loc="upper left")
    ax[1].set_ylabel(r"exponential rate $\alpha$ [1/s]")
    ax[1].set_title("local exponential rate: theory vs measured", fontsize=9)
    ax[2].set_ylabel(r"transient constant $\hat c$  (median, IQR)")
    ax[2].set_yscale("log")
    ax[2].set_title(r"overshoot / saturation: $e(t) \leq \hat c\, e(0)\, e^{-\hat\alpha t}$",
                    fontsize=9)
    fig.tight_layout()
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
                       color="white" if v > 60 else style.TEXT)
        a.set_xticks(range(len(radii))); a.set_xticklabels([f"{r:.2f}" for r in radii])
        a.set_yticks(range(len(dicts))); a.set_yticklabels([style.LABEL[n] for n in names], fontsize=8)
        a.set_xlabel("training region  r_train [rad]")
        a.set_title(f"{'infinite-horizon LQR' if meth == 'inf' else 'receding-horizon gain, 0.5 s'}"
                    f"  (local LQR: {100 * loc[meth]:.0f}%)", fontsize=9)
        a.grid(False)
        fig.colorbar(im, ax=a, label="validation success [%]")
    fig.suptitle("Sensitivity: success on 400 random validation starts", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/scan_sensitivity.png"); plt.close(fig)


def summary_table(names):
    design = json.load(open(f"{RES}/design_summary.json"))
    roa = json.load(open(f"{RES}/roa_summary.json"))
    base = load_map("local")
    rows = []
    for n in ["local"] + cfg.CONTROL_DICTS:
        d = design[n]
        row = dict(controller=style.LABEL[n], n_obs=d["n_obs"], suitable=d["suitable"],
                   alpha_model=d.get("alpha_model", np.nan), alpha_true_local=d.get("alpha_true", np.nan))
        if n in names:
            r = load_map(n); s = r["success"]
            both = s & base["success"]
            row.update(
                success_pct=100 * s.mean(),
                certified_pct=100 * roa[n]["area_frac"],
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


def stage_figures():
    names = suitable_names()
    fig_success_maps(names)
    fig_who_wins(names)
    fig_roa(names)
    best, pick = fig_timeseries(names)
    fig_convergence(names, pick)
    if os.path.exists(f"{RES}/scan.csv"):
        fig_scan()
    summary_table(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "fit", "design", "maps", "roa", "figures"])
    ap.add_argument("--fast", action="store_true", help="vectorized maps (ideal only)")
    a = ap.parse_args()
    if a.stage in ("all", "fit"): stage_fit()
    if a.stage in ("all", "design"): stage_design()
    if a.stage in ("all", "maps"): stage_maps(a.fast)
    if a.stage in ("all", "roa"): stage_roa()
    if a.stage in ("all", "figures"): stage_figures()


if __name__ == "__main__":
    main()
