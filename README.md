# Koopman-Based and Local-Linearization LQR Comparison

ME627 class research project using the DFKI double-pendulum framework.

## Goal

Compare LQR designed using a locally linearized model with LQR
designed using a data-driven Koopman lifted model.

Both controllers will be evaluated on the same nonlinear,
fully actuated two-link system. The project will investigate
how performance changes with initial conditions and lifting features.

## Framework

- Simulation: DFKI Double Pendulum
- Language: Python
- Conda environment: ece627
- State: two joint angles and two joint velocities
- Input: two joint torques

Upstream repository:
https://github.com/dfki-ric-underactuated-lab/double_pendulum

The DFKI repository is maintained separately from this project.

## Setup

Activate the existing environment:

    conda activate ece627

Install the DFKI Python package once into that environment,
replacing the path below with your local checkout:

    python -m pip install -e "PATH_TO_DFKI/src/python"

Verify the import:

    python -c "from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum; print('Import OK')"

## Current Progress

- [x] Selected the DFKI simulation framework.
- [x] Confirmed that DFKI imports work from this repository.
- [x] Verify the sample simulation and animation.
- [x] Define plant parameters and target equilibrium (`config.py`).
- [x] Implement the local-linearization LQR baseline.
- [x] Collect trajectories and identify Koopman models.
- [x] Compare closed-loop performance (maps, rates, regions of attraction).
- [ ] Realistic-sensing repeat, report and presentation.

## Running

    python run_local_lqr.py --map     # baseline: design, tests, figures, map
    python collect_data.py            # identification data -> results/ident_data.npz
    python scan_designs.py            # design-choice scan on validation starts (~2 min)
    python run_koopman.py             # fit, design, DFKI maps, ROA, figures, table
    python run_koopman.py --fast      # same, maps with the vectorized sweep
    python hardware.py                # budget-rig plant parameters

`run_koopman.py --stage fit|design|maps|roa|figures` runs one stage.

## Code map

| File | Role |
|---|---|
| `config.py` | every parameter: plant, target, dt, Q/R, torque limit, data, dictionaries |
| `plant_utils.py` | DFKI plant + simulator, equilibrium, Jacobians, ZOH discretization |
| `dynamics_np.py` | vectorized copy of DFKI's dynamics (checked to ~1e-13) |
| `lqr_design.py` | discrete LQR with stabilizability / detectability (PBH) checks |
| `controllers.py` | one controller for all methods: `u = sat(u* - K psi(x - x*))` |
| `metrics.py` | success, settling, RMSE, cost J, effort, saturation, tail rate, c_hat |
| `collect_data.py` | Stage 3 data, segmented at angle wraps, split by trajectory |
| `koopman.py` | dictionaries (polynomial, trig), EDMDc fit, multistep prediction |
| `stability.py` | Koopman LQR design, true-plant local rates, sampled ROA certificate |
| `scan_designs.py` | training region x dictionary x gain design, on validation starts |
| `sweep.py` / `fast_sweep.py` | initial-angle maps (DFKI / vectorized) |
| `run_koopman.py` | Stages 4-5 pipeline and all comparison figures |
| `viz.py`, `style.py` | figure helpers and the shared colour/line-style per controller |

## Experiments and Results

See `results/summary_ideal-ideal.md` and `figures/`.  Key design facts:

* Infinite-horizon LQR on a polynomial lifted model is ill-posed at an
  unstable equilibrium: products of unstable modes are unstable lifted
  modes with vanishing control authority, so the Riccati gain blows up.
* Wide-region least squares fits gravity with a secant, softer than the
  tangent at upright, so wide-data models can fail to stabilize locally.
* The Koopman controller's local behaviour is set by `K dpsi/dx(0)`; its
  stability guarantee is for the model, the true plant is checked
  separately (`stability.true_local_eigs`).

## Reproducibility

DFKI commit: to be recorded.

A verified environment specification will be added later.