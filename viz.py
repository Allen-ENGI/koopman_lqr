"""All figures for the project (headless-safe: everything is saved to files)."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap, LogNorm

import config as cfg
from plant_utils import state_error


def _trim(T, X, U):
    n = min(len(T), len(X), len(U))
    return np.asarray(T)[:n], np.asarray(X)[:n], np.asarray(U)[:n]


# ------------------------------------------------ 1. time series
def plot_run(T, X, U, title, path, x_star=cfg.X_STAR):
    """Angle errors, velocities and torques vs time (figure type 1)."""
    T, X, U = _trim(T, X, U)
    dX = np.array([state_error(x, x_star) for x in X])
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(T, dX[:, 0], label=r"$q_1-q_1^\star$")
    ax[0].plot(T, dX[:, 1], label=r"$q_2-q_2^\star$")
    ax[0].axhspan(-cfg.ANGLE_TOL, cfg.ANGLE_TOL, color="g", alpha=0.12, lw=0)
    ax[0].set_ylabel("angle error [rad]")
    ax[1].plot(T, X[:, 2], label=r"$\dot q_1$")
    ax[1].plot(T, X[:, 3], label=r"$\dot q_2$")
    ax[1].set_ylabel("velocity [rad/s]")
    for j, c in enumerate(["C0", "C1"]):
        ax[2].plot(T, U[:, j], c, label=f"$u_{j+1}$")
        for s in (1, -1):
            ax[2].axhline(s * cfg.TORQUE_LIMIT[j], color=c, ls=":", lw=0.8)
    ax[2].set_ylabel("torque [N m]")
    ax[2].set_xlabel("time [s]")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overlay(runs, title, path, x_star=cfg.X_STAR):
    """Several runs on one figure: angle-error norm and both torques.
    runs = [(label, T, X, U), ...]"""
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for i, (label, T, X, U) in enumerate(runs):
        T, X, U = _trim(T, X, U)
        dX = np.array([state_error(x, x_star) for x in X])
        c = f"C{i}"
        ax[0].semilogy(T, np.linalg.norm(dX[:, :2], axis=1) + 1e-6, c, label=label)
        ax[1].plot(T, U[:, 0], c)
        ax[2].plot(T, U[:, 1], c)
    ax[0].axhline(cfg.ANGLE_TOL, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel(r"$\|q-q^\star\|$ [rad]")
    for k, j in ((1, 0), (2, 1)):
        for s in (1, -1):
            ax[k].axhline(s * cfg.TORQUE_LIMIT[j], color="k", ls=":", lw=0.8)
        ax[k].set_ylabel(f"$u_{j+1}$ [N m]")
    ax[2].set_xlabel("time [s]")
    ax[0].legend(fontsize=8, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------ animation
def save_animation(T, X, U, path, title="", fps=30, x_star=cfg.X_STAR):
    """Pendulum animation with a trail and torque bars, saved as GIF/MP4."""
    T, X, U = _trim(T, X, U)
    l1, l2 = cfg.LENGTH
    step = max(1, int(round(1.0 / (fps * (T[1] - T[0])))))
    idx = np.arange(0, len(T), step)

    def joints(q):
        p1 = np.array([l1 * np.sin(q[0]), -l1 * np.cos(q[0])])
        p2 = p1 + np.array([l2 * np.sin(q[0] + q[1]), -l2 * np.cos(q[0] + q[1])])
        return p1, p2

    fig = plt.figure(figsize=(8, 4.2))
    axp = fig.add_axes([0.03, 0.08, 0.5, 0.84])
    axu = fig.add_axes([0.62, 0.15, 0.34, 0.7])
    L = 1.15 * (l1 + l2)
    axp.set_xlim(-L, L); axp.set_ylim(-L, L); axp.set_aspect("equal")
    axp.axis("off")
    g1, g2 = joints(x_star[:2])
    axp.plot([0, g1[0], g2[0]], [0, g1[1], g2[1]], color="0.8", lw=4, zorder=0)
    arm, = axp.plot([], [], "o-", lw=3, ms=7, color="C0")
    trail, = axp.plot([], [], "-", lw=1, color="C1", alpha=0.6)
    txt = axp.text(-L, L * 0.9, "", fontsize=9)
    axp.set_title(title, fontsize=10)
    lim = np.asarray(cfg.TORQUE_LIMIT)
    bars = axu.bar(["$u_1/u_{1,max}$", "$u_2/u_{2,max}$"], [0, 0], color=["C0", "C1"])
    axu.set_ylim(-1.05, 1.05); axu.axhline(0, color="k", lw=0.5)
    axu.set_title("normalised torque", fontsize=10)
    tip_hist = []

    def update(k):
        i = idx[k]
        p1, p2 = joints(X[i, :2])
        arm.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])
        tip_hist.append(p2)
        th = np.array(tip_hist[-60:])
        trail.set_data(th[:, 0], th[:, 1])
        txt.set_text(f"t = {T[i]:.2f} s")
        for b, v in zip(bars, U[i] / lim):
            b.set_height(v)
            b.set_color("C3" if abs(v) > 0.999 else b.get_facecolor())
        return arm, trail, txt, *bars

    anim = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
    writer = (animation.PillowWriter(fps=fps) if str(path).endswith(".gif")
              else animation.FFMpegWriter(fps=fps))
    anim.save(path, writer=writer)
    plt.close(fig)


# ------------------------------------------------ angle conventions (report)
def draw_conventions(path):
    """Sketch of the DFKI angle conventions and the chosen target."""
    l1, l2 = cfg.LENGTH
    q1, q2 = 0.6, 0.8
    p1 = np.array([l1 * np.sin(q1), -l1 * np.cos(q1)])
    p2 = p1 + np.array([l2 * np.sin(q1 + q2), -l2 * np.cos(q1 + q2)])
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    a = ax[0]
    a.plot([0, 0], [0, -1.1 * l1], "k--", lw=0.8)
    a.plot([p1[0], p1[0] + 0.8 * l2 * np.sin(q1)],
           [p1[1], p1[1] - 0.8 * l2 * np.cos(q1)], "k--", lw=0.8)
    a.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], "o-", lw=3, color="C0")
    t = np.linspace(0, q1, 30)
    a.plot(0.35 * l1 * np.sin(t), -0.35 * l1 * np.cos(t), "C3")
    a.text(0.42 * l1 * np.sin(q1 / 2), -0.42 * l1 * np.cos(q1 / 2), "$q_1$", color="C3")
    t = np.linspace(q1, q1 + q2, 30)
    a.plot(p1[0] + 0.4 * l2 * np.sin(t), p1[1] - 0.4 * l2 * np.cos(t), "C3")
    a.text(p1[0] + 0.5 * l2 * np.sin(q1 + q2 / 2),
           p1[1] - 0.5 * l2 * np.cos(q1 + q2 / 2), "$q_2$ (relative)", color="C3")
    a.annotate("", xy=(0, -0.1), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    a.set_title("$q_1=0$: link 1 hangs down; CCW positive")
    a.set_aspect("equal"); a.axis("off")
    b = ax[1]
    b.plot([0, 0, 0], [0, l1, l1 + l2], "o-", lw=3, color="C2")
    b.set_title(r"target $x^\star=[\pi,0,0,0]$ (upright), $u^\star=0$")
    b.set_xlim(-0.3, 0.3); b.set_aspect("equal"); b.axis("off")
    for a in ax:
        a.plot(0, 0, "ks", ms=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------ 3. initial-angle maps
def plot_map(res, title, path):
    """Success / cost / settling time over the (dq1, dq2) slice.
    res: dict from sweep.run_map (grid axis + 2-D metric arrays)."""
    g = res["grid"]
    ext = [g[0], g[-1], g[0], g[-1]]
    ok = res["success"].astype(float)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].imshow(ok, origin="lower", extent=ext,
                 cmap=ListedColormap(["#d9534f", "#5cb85c"]), vmin=0, vmax=1)
    ax[0].set_title(f"success ({100 * ok.mean():.0f}% of slice)")
    J = np.where(res["success"], res["cost_J"], np.nan)
    # robust log scale: the run starting at x* has J ~ 0 and would stretch it
    Jpos = J[np.isfinite(J) & (J > 0)]
    norm = (LogNorm(vmin=max(np.percentile(Jpos, 1), 1e-3), vmax=Jpos.max(), clip=True)
            if Jpos.size else None)
    im = ax[1].imshow(J, origin="lower", extent=ext, cmap="viridis", norm=norm)
    fig.colorbar(im, ax=ax[1], label="J (successful runs)")
    ax[1].set_title("common quadratic cost")
    ts = np.where(res["success"], res["settling_time"], np.nan)
    im = ax[2].imshow(ts, origin="lower", extent=ext, cmap="magma_r")
    fig.colorbar(im, ax=ax[2], label="s")
    ax[2].set_title("settling time")
    for a in ax:
        a.contour(g, g, ok, levels=[0.5], colors="k", linewidths=1)
        a.plot(0, 0, "w+", ms=12, mew=2)
        a.set_xlabel(r"$\Delta q_1(0)$ [rad]")
        a.set_ylabel(r"$\Delta q_2(0)$ [rad]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_map_comparison(res_a, res_b, name_a, name_b, path):
    """Who wins on the slice (figure type 3, used once Koopman exists).
    0 both fail, 1 only A, 2 only B, 3 both succeed and A cheaper, 4 both and B cheaper."""
    sa, sb = res_a["success"], res_b["success"]
    cls = np.zeros(sa.shape, int)
    cls[sa & ~sb] = 1
    cls[~sa & sb] = 2
    both = sa & sb
    cls[both & (res_a["cost_J"] <= res_b["cost_J"])] = 3
    cls[both & (res_a["cost_J"] > res_b["cost_J"])] = 4
    colors = ["#444444", "#1f77b4", "#ff7f0e", "#aec7e8", "#ffbb78"]
    labels = ["both fail", f"only {name_a}", f"only {name_b}",
              f"both, {name_a} cheaper", f"both, {name_b} cheaper"]
    g = res_a["grid"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cls, origin="lower", extent=[g[0], g[-1], g[0], g[-1]],
              cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5)
    for c, l in zip(colors, labels):
        ax.plot([], [], "s", color=c, ms=10, label=f"{l} ({np.mean(cls == labels.index(l)) * 100:.0f}%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_xlabel(r"$\Delta q_1(0)$ [rad]"); ax.set_ylabel(r"$\Delta q_2(0)$ [rad]")
    ax.set_title(f"{name_a} vs {name_b}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)