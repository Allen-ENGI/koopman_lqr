"""Single source of truth for the plant, target, sampling and cost.

Every controller (local-linearization LQR and all Koopman LQRs) must
import from here so that the comparison is fair.

Two independent switches:
  PLANT_PROFILE  "ideal"     : clean study plant (point masses at the tips)
                 "hardware"  : the planned budget rig (hardware.py):
                               GM5208-12 shoulder + GM3506 elbow, direct drive,
                               3D-printed 12 / 10 cm links
  REALISM        "ideal"     : perfect state, no delay, no Coulomb friction
                 "realistic" : friction, encoder/velocity noise, 1-step delay,
                               finite motor bandwidth (what the real rig will do)

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

PLANT_PROFILE = "ideal"          # "ideal" | "hardware"
REALISM = "ideal"                # "ideal" | "realistic"

GRAVITY = 9.81

# ---------------------------------------------------------------- plant
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
    # Static gravity demand (arm horizontal) ~[4.2, 1.0] N m.  Local-LQR
    # success on the zero-velocity slice (15x15 sweep):
    #   [8, 2] 100%  [5, 1.3] 100%  [4.4, 1.1] 100%  [4, 1] 91%  [3.6, 0.9] 46%
    # Above the static demand the slice is trivially 100%, so the study limit
    # sits just below it to put the local-LQR boundary inside the map.
    TORQUE_LIMIT = [4.0, 1.0]    # N m, the limit used in ALL experiments
elif PLANT_PROFILE == "hardware":
    import hardware as _hw
    _p = _hw.build(torque_mode="rated")
    MASS, LENGTH, COM, INERTIA = _p["MASS"], _p["LENGTH"], _p["COM"], _p["INERTIA"]
    MOTOR_INERTIA, GEAR_RATIO = _p["MOTOR_INERTIA"], _p["GEAR_RATIO"]
    # viscous damping placeholder, ~1% of rated torque per rad/s (identify!)
    DAMPING = list(0.01 * np.asarray(_p["TORQUE_LIMIT"]))
    MOTOR_TORQUE_LIMIT = _p["TORQUE_LIMIT"]
    # Software limit for the study.  As in the ideal profile, the
    # interesting regime puts the SHOULDER limit just below its static
    # gravity demand.  The elbow demand is tiny (cogging-level), so the elbow
    # keeps its full rated torque -- which lets the arm fold and shorten the
    # lever, so the shoulder must sit further below its static demand.
    # Local-LQR success on the slice (GM5208 + GM3506 rig, 15x15):
    #   ratio 0.85: 100% / 100%   0.75: 75% / 32%   0.65: 24% / 24%
    #   0.80 (25x25): 100% / 50%   (ideal / realistic).
    # Realism matters a lot here; re-tune with run_local_lqr.py --map.
    _need = _hw.static_torque_demand(_p)
    TORQUE_LIMIT = [min(0.80 * _need[0], MOTOR_TORQUE_LIMIT[0]), MOTOR_TORQUE_LIMIT[1]]
else:
    raise ValueError(PLANT_PROFILE)

assert np.all(np.asarray(TORQUE_LIMIT) <= np.asarray(MOTOR_TORQUE_LIMIT))

# ------------------------------------------------------- non-idealities
if REALISM == "ideal":
    COULOMB_FRIC = [0.0, 0.0]          # N m
    MEAS_NOISE = [0.0, 0.0, 0.0, 0.0]  # std of [q1, q2, dq1, dq2]
    MEAS_DELAY = 0.0                   # s
    TORQUE_NOISE = [0.0, 0.0]          # N m std
    MOTOR_RESPONSIVENESS = 1.0         # 1 = torque applied instantly
elif REALISM == "realistic":
    # Coulomb friction ~2.5% of the motor rating (order of magnitude from
    # DFKI's identified bench); 14-bit AS5048A encoders with numerically
    # differentiated velocity.
    COULOMB_FRIC = list(0.025 * np.minimum(np.asarray(MOTOR_TORQUE_LIMIT), 4.0))
    MEAS_NOISE = [0.001, 0.001, 0.05, 0.05]
    MEAS_DELAY = 0.005                 # one control period
    TORQUE_NOISE = list(0.005 * np.minimum(np.asarray(MOTOR_TORQUE_LIMIT), 4.0))  # 0.5% of rating
    MOTOR_RESPONSIVENESS = 0.8
else:
    raise ValueError(REALISM)

# --------------------------------------------------------------- target
X_STAR = np.array([np.pi, 0.0, 0.0, 0.0])   # both links upright
# U_STAR is computed from the plant (0 for upright, but not assumed).

# ------------------------------------------------------------- sampling
DT = 0.005                   # s, 200 Hz: control period = RK4 step in DFKI
T_FINAL = 5.0                # s
INTEGRATOR = "runge_kutta"

# --------------------------------- physical cost (shared by ALL controllers)
Q = np.diag([10.0, 10.0, 1.0, 1.0])
R = np.diag([1.0, 1.0])

# ------------------------------------------------------- success criterion
ANGLE_TOL = 0.05             # rad
VEL_TOL = 0.2                # rad / s
HOLD_TIME = 0.5              # s the state must stay inside the tolerances
VEL_AVG_WINDOW = 0.1        # s, velocity averaging for the "at rest" test

# ------------------------------------------------------- initial-angle map
MAP_N = 41                   # grid points per axis
MAP_RANGE = np.pi            # dq1, dq2 in [-MAP_RANGE, MAP_RANGE]
SEED = 0

# ------------------------------------------------ Koopman identification
# Data: whole trajectories, split 60/20/20 by trajectory (never by sample).
# Two kinds of trajectory, both starting across the whole map slice:
#   closed: local LQR + random exploration torque (covers the region the
#           controllers actually visit, incl. near-upright behaviour)
#   open:   SHORT open-loop random torque bursts (falling behaviour near each
#           start, without long swings around hanging-down that would
#           dominate the least-squares fit)
DATA_N_CLOSED, DATA_T_CLOSED = 600, 1.5      # trajectories, s
DATA_N_OPEN, DATA_T_OPEN = 1200, 0.4
DATA_V0 = 3.0                # initial velocities ~ U(-DATA_V0, DATA_V0) rad/s
DATA_V_MAX = 15.0            # drop samples faster than this (rad/s)
DATA_HOLD = (0.02, 0.10)     # s, random hold time of piecewise-constant inputs
DATA_SPLIT = (0.6, 0.2, 0.2)
# Observable dictionaries (names used everywhere):
#   poly<d>: monomials of the scaled error up to degree d
#            (poly1 = state only = DMDc, poly2 = quadratic, poly3 = cubic)
#   trig<l>: physics-matched sin / cos observables (extension, koopman.TrigLift)
PRED_DICTS = ["poly1", "poly2", "poly3", "poly4", "trig1", "trig2"]
CONTROL_DICTS = ["poly1", "poly2", "poly3", "trig1", "trig2"]
# Training region for the MAIN experiment: pairs with |dq1|,|dq2| <= R.
# Same data for every dictionary.  Chosen on validation starts by
# scan_designs.py (best for state-only and trig1; see results/scan.csv).
KOOPMAN_R_TRAIN = 0.6        # rad
LQR_METHOD = "inf"           # same gain design for every model
RIDGE_GRID = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]   # relative ridge weights
PRED_HORIZON = 0.25          # s, horizon used to pick the ridge weight
                             # (~2 time constants of the unstable modes;
                             #  errors grow ~e^(7.6 t) open loop)
PRED_PLOT_HORIZON = 0.5      # s, horizon shown in the prediction figures
LIFTED_Q_EPS = 1e-6          # added on lifted states only if detectability fails

# ------------------------------------------------ stability analysis
RATE_E_HI = 0.5              # tail-rate fit window on sqrt(dx'Q dx)
RATE_E_LO = 1e-4
ROA_SAMPLES = 3_000_000      # 4-D samples for the sampled Lyapunov certificate
ROA_V_BOX = 40.0             # rad/s, velocity half-width of the sampled box
