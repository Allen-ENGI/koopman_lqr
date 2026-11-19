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

The comparison is two simulations over the same conditions. Every variable
that is changed lives in `config.py`, section 1 (EXPERIMENT VARIABLES):

| Variable | Balance (Exp. 1, 2) | Tracking (Exp. 3) | Config name |
|---|---|---|---|
| Initial angles (dq1, dq2) | main: grid inside every condition, also grouped by distance from upright | fixed: starts on the letter | `MAP_N`, `MAP_RANGE`, `DISTANCE_EDGES` |
| Observables | main: one Koopman controller per set | main: one Koopman controller per set | `OBSERVABLES`, `TRACK_OBSERVABLES` |
| Region size | the distance bands of the map | two scenes: small letter near upright, large letters around the shoulder | `DISTANCE_EDGES`, `TRACK_SCENES`, `TRACK_RUN` |
| Model error (physics designs only) | robustness (default) | robustness (default) | `MODEL_ERROR_SWEEP`, `MODEL_ERROR_LINKS` ("elbow" = wrong mass ratio; "both" = only rescales the gain), `ROBUSTNESS`, `TRACK_ROBUSTNESS` |
| Torque limit, initial velocity, plant change, sensor noise | optional robustness | optional (plant, noise) | add to `ROBUSTNESS` / `TRACK_ROBUSTNESS` |

Robustness changes one variable at a time; the others stay at their main value.
By default only model error runs: it is the robustness test that asks the
research question (does data beat a wrong physics model?). Everything else in
`config.py` is held fixed for both controllers.

    python plant.py                        # check: vectorized plant matches DFKI
    python run_local_lqr.py --maps all     # SIMULATION 1: local LQR, main + ROBUSTNESS
    python run_koopman.py                  # SIMULATION 2: data, fit, design, Koopman maps,
                                           #   comparison figures (incl. by_distance.png)
    python run_tracking.py --check         # Experiment 3: are the letters feasible?
    python run_tracking.py                 # Experiment 3: both scenes, all controllers,
                                           #   figures, GIFs, summary_scenes.md
    python run_koopman.py --stage extra    # optional: design scan, Lyapunov certificates,
                                           #   convergence rates

`--maps main|robust|all` picks the conditions; `run_koopman.py --stage
data|fit|design|maps|figures` runs one stage. Balance maps are saved as
`results/maps/<profile>-<realism>/<controller>__<condition>.npz`, so the
simulations line up condition by condition. `MAIN_SIMULATOR` and
`ROBUSTNESS_SIMULATOR` choose DFKI's simulator or its vectorized copy.

## Code map

| File | Role |
|---|---|
| `config.py` | every parameter; section 1 = the experiment variables |
| `plant.py` | DFKI plant and simulator, linearization, ZOH discretization, vectorized copy with inverse dynamics |
| `tool_lib.py` | conditions, one-run simulation, initial-angle maps, metrics, tracking references, figure style and plotting |
| `koopman.py` | identification data, observables (polynomial, trig), lifted least-squares fit, multistep prediction |
| `control.py` | every controller design: LQR with PBH checks, local LQR, Koopman LQR, stability analysis, TVLQR, tracking controllers |
| `run_local_lqr.py` | SIMULATION 1 (balance): local LQR |
| `run_koopman.py` | SIMULATION 2 (balance): Koopman LQR and all comparison figures |
| `run_tracking.py` | Experiment 3: drawing letters, small scene near upright and large scene around the shoulder (figures, GIFs, tables) |
| `hardware.py` | parameters of the planned budget rig (`PLANT_PROFILE = "hardware"`) |

Dependency order (no cycles): `config` -> `plant` -> `tool_lib` -> `koopman` -> `control` -> the three `run_*` scripts.

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