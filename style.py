"""Shared figure style: one fixed colour per controller (never by rank),
plus a line style and marker so identity never relies on colour alone.
Categorical palette validated for colour-vision deficiency (CVD) separation
(dataviz validator: all checks pass; 3 slots < 3:1 contrast on white, so
every figure also uses line style / markers / legends)."""
import matplotlib as mpl

TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GOOD, CRITICAL, NEUTRAL = "#0ca30c", "#d03b3b", "#bdbcb6"

ORDER = ["local", "poly1", "poly2", "poly3", "trig1", "trig2"]
COLOR = {"local": "#2a78d6", "poly1": "#eb6834", "poly2": "#1baf7a",
         "poly3": "#eda100", "trig1": "#e87ba4", "trig2": "#6250d6",
         "poly4": "#8a8984", "taylor": "#2a78d6"}
LS = {"local": "-", "poly1": "--", "poly2": "-.", "poly3": ":",
      "trig1": "-", "trig2": "--", "poly4": ":", "taylor": "-"}
MARKER = {"local": "o", "poly1": "s", "poly2": "^", "poly3": "v",
          "trig1": "D", "trig2": "P", "poly4": "x", "taylor": "o"}
LABEL = {"local": "local LQR (Taylor)", "taylor": "Taylor linearization",
         "poly1": "Koopman: state only", "poly2": "Koopman: quadratic",
         "poly3": "Koopman: cubic", "poly4": "Koopman: quartic",
         "trig1": "Koopman: trig (8)", "trig2": "Koopman: trig (13)"}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "lines.linewidth": 2.0, "lines.markersize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })
