"""Single source of truth for the comparison.

Section 1 holds the EXPERIMENT VARIABLES: the only values you change to
compare the two controllers.  Both simulations read them:
    run_local_lqr.py   local-linearization LQR
    run_koopman.py     Koopman LQR (one controller per observable set)
Everything after section 1 is held FIXED across every comparison.

DFKI conventions (SymbolicDoublePendulum, formulas="UnderactuatedLecture"):
  * state  x = [q1, q2, dq1, dq2]
  * q1 = 0      -> link 1 hangs straight DOWN (potential = -r1*cos(q1))
  * q2          -> angle of link 2 RELATIVE to link 1
  * positive angles are counter-clockwise; tip 1 = (l1 sin q1, -l1 cos q1)
  * DFKI writes  M qdd + C qd - G + F = u,  i.e. DFKI's G = -g(q) in
    the notation M qdd + C qd + g(q) + F(qd) = u
  * F(qd) = b * qd + cf * atan(100 qd)
  * `inertia` is the link inertia about its JOINT axis
"""
import numpy as np

# =====================================================================
# 1. EXPERIMENT VARIABLES  (change these; both simulations follow)
# =====================================================================
# Every run starts at   x0 = x* + [dq1, dq2, v1, v2]   and is scored on
# whether it reaches x* (success) and at what cost J.  For a given
# condition, the two simulations differ ONLY in the controller.

# --- (a) Initial angles: Experiment 1 -------------------------------
# (dq1, dq2) on an N x N grid; every controller runs from every point.
MAP_N = 31                   # grid points per axis -> 31 x 31 = 961 starts
MAP_RANGE = np.pi            # dq1, dq2 in [-MAP_RANGE, MAP_RANGE] rad

# --- (b) Observables: Experiment 2 ------------------------------------
# One Koopman controller per observable set z = psi(dx) = [dx ; extras]:
#   "poly1"  state only (= DMDc)                               N = 4
#   "poly2"  + all degree-2 products of the scaled error       N = 14
#   "poly3"  + all degree-3 products                           N = 34
#   "trig1"  + sin dq1, sin(dq1+dq2), sin dq2, 1 - cos dq2     N = 8
#   "trig2"  + trig1 + sin dq2 * qd_i qd_j, (1 - cos dq2) dq_i N = 13
# Local LQR has no observables (psi = dx, Taylor model).
OBSERVABLES = ["poly1", "poly2", "poly3", "trig1", "trig2"]

# --- (c) Main condition (used by Experiments 1 and 2) -----------------
TORQUE_LIMIT_MAIN = [4.0, 1.0]   # N m; ideal plant only (holding torque is ~4.2 N m)
INIT_VELOCITY_MAIN = (0.0, 0.0)  # rad/s, (qd1, qd2) at t = 0

# --- (d) Robustness: change ONE variable at a time, rest at main ------
TORQUE_SCALE_SWEEP = [0.9, 1.1, 1.25]   # x the main torque limit
                                        # -> [3.6, 0.9], [4.4, 1.1], [5.0, 1.25] N m
INIT_VELOCITY_SWEEP = [(2.0, 0.0), (-2.0, 0.0), (0.0, 3.0), (3.0, -3.0)]  # rad/s

# --- Which simulator runs the maps ------------------------------------
# "dfki": DFKI's own Simulator (any REALISM, ~3.5 min per 31 x 31 map on 2 cores)
# "fast": vectorized copy of the same plant (matches DFKI to ~1e-13, ~30x
#         faster, ideal sensing only; falls back to "dfki" otherwise)
MAIN_SIMULATOR = "dfki"
ROBUSTNESS_SIMULATOR = "fast"

# --- Plant and sensing switches -----------------------------------------
PLANT_PROFILE = "ideal"      # "ideal" (study plant) | "hardware" (budget rig, hardware.py)
REALISM = "ideal"            # "ideal" | "realistic" (friction, noise, 5 ms delay)

# =====================================================================
# 2. FIXED for every comparison (same for both controllers)
# =====================================================================
X_STAR = np.array([np.pi, 0.0, 0.0, 0.0])   # target: both links upright
DT = 0.005                   # s, 200 Hz: control period = RK4 step in DFKI
T_FINAL = 5.0                # s per run
INTEGRATOR = "runge_kutta"

# Physical cost: used by BOTH designs and to score every trajectory
Q = np.diag([10.0, 10.0, 1.0, 1.0])
R = np.diag([1.0, 1.0])

# Success: both angle errors < ANGLE_TOL and both velocities (averaged over
# VEL_AVG_WINDOW) < VEL_TOL, held for HOLD_TIME before T_FINAL
ANGLE_TOL = 0.05             # rad
VEL_TOL = 0.2                # rad / s
HOLD_TIME = 0.5              # s
VEL_AVG_WINDOW = 0.1         # s
COST_TIE = 0.01              # |J_a / J_b - 1| below this counts as a tie

SEED = 0
GRAVITY = 9.81

# =====================================================================
# 3. Plant (fixed per profile)
# =====================================================================
if PLANT_PROFILE == "ideal":
    # Point masses at the link tips: com = length, inertia = m * l^2.
    MASS = [0.6, 0.5]            # kg
    LENGTH = [0.3, 0.2]          # m
    COM = [0.3, 0.2]             # m
    INERTIA = [MASS[0] * COM[0] ** 2, MASS[1] * COM[1] ** 2]
    MOTOR_INERTIA = 0.0
    GEAR_RATIO = 1
    DAMPING = [0.05, 0.05]       # N m s / rad
    MOTOR_TORQUE_LIMIT = [np.inf, np.inf]
    TORQUE_LIMIT = list(TORQUE_LIMIT_MAIN)
elif PLANT_PROFILE == "hardware":
    import hardware as _hw
    _p = _hw.build(torque_mode="rated")
    MASS, LENGTH, COM, INERTIA = _p["MASS"], _p["LENGTH"], _p["COM"], _p["INERTIA"]
    MOTOR_INERTIA, GEAR_RATIO = _p["MOTOR_INERTIA"], _p["GEAR_RATIO"]
    # viscous damping placeholder, ~1% of rated torque per rad/s (identify!)
    DAMPING = list(0.01 * np.asarray(_p["TORQUE_LIMIT"]))
    MOTOR_TORQUE_LIMIT = _p["TORQUE_LIMIT"]
    # Study limit: shoulder at 0.80 x its static gravity demand, elbow at
    # its full rating (local-LQR success 100% ideal / 50% realistic).
    _need = _hw.static_torque_demand(_p)
    TORQUE_LIMIT = [min(0.80 * _need[0], MOTOR_TORQUE_LIMIT[0]), MOTOR_TORQUE_LIMIT[1]]
else:
    raise ValueError(PLANT_PROFILE)

assert np.all(np.asarray(TORQUE_LIMIT) <= np.asarray(MOTOR_TORQUE_LIMIT))

# --------------------------------------------------------- sensing / actuation
if REALISM == "ideal":
    COULOMB_FRIC = [0.0, 0.0]          # N m
    MEAS_NOISE = [0.0, 0.0, 0.0, 0.0]  # std of [q1, q2, dq1, dq2]
    MEAS_DELAY = 0.0                   # s
    TORQUE_NOISE = [0.0, 0.0]          # N m std
    MOTOR_RESPONSIVENESS = 1.0         # 1 = torque applied instantly
elif REALISM == "realistic":
    # Coulomb friction ~2.5% of the motor rating; 14-bit encoders with
    # numerically differentiated velocity; one control period of delay.
    COULOMB_FRIC = list(0.025 * np.minimum(np.asarray(MOTOR_TORQUE_LIMIT), 4.0))
    MEAS_NOISE = [0.001, 0.001, 0.05, 0.05]
    MEAS_DELAY = 0.005
    TORQUE_NOISE = list(0.005 * np.minimum(np.asarray(MOTOR_TORQUE_LIMIT), 4.0))
    MOTOR_RESPONSIVENESS = 0.8
else:
    raise ValueError(REALISM)

# =====================================================================
# 4. Koopman identification (fixed; same data for every observable set)
# =====================================================================
# Data: whole trajectories, split 60/20/20 by trajectory (never by sample).
#   closed: local LQR + random exploration torque (region the controllers visit)
#   open:   SHORT open-loop random torque bursts (falling behaviour)
DATA_N_CLOSED, DATA_T_CLOSED = 600, 1.5      # trajectories, s
DATA_N_OPEN, DATA_T_OPEN = 1200, 0.4
DATA_V0 = 3.0                # initial velocities ~ U(-DATA_V0, DATA_V0) rad/s
DATA_V_MAX = 15.0            # drop samples faster than this (rad/s)
DATA_HOLD = (0.02, 0.10)     # s, random hold time of piecewise-constant inputs
DATA_SPLIT = (0.6, 0.2, 0.2)
# Fit only pairs with |dq1|, |dq2| <= KOOPMAN_R_TRAIN (chosen on validation
# starts by scan_designs.py)
KOOPMAN_R_TRAIN = 0.6        # rad
LQR_METHOD = "inf"           # same gain design for every model
RIDGE_GRID = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]   # relative ridge weights
PRED_HORIZON = 0.25          # s, horizon used to pick the ridge weight
PRED_PLOT_HORIZON = 0.5      # s, horizon shown in the prediction figure
PRED_DICTS = OBSERVABLES + ["poly4"]          # quartic: prediction only
CONTROL_DICTS = OBSERVABLES                    # older name, kept for scripts
LIFTED_Q_EPS = 1e-6          # added on lifted states only if detectability fails

# =====================================================================
# 5. Analysis settings
# =====================================================================
RATE_E_HI = 0.5              # tail-rate fit window on sqrt(dx'Q dx)
RATE_E_LO = 1e-4
ROA_SAMPLES = 3_000_000      # 4-D samples for the sampled Lyapunov certificate
ROA_V_BOX = 40.0             # rad/s, velocity half-width of the sampled box
