# Koopman vs Local LQR: Code Walkthrough

Nov 18, 2026 · exported from the shared walkthrough doc

The project is 9 Python files: five shared modules, three scripts you run, and `hardware.py` for the future rig. The main balance experiments (stages 1–5) are complete. The evaluation is now trimmed to one question, small region vs large region, plus one robustness test, model error. The new distance figure, the two tracking scenes and model error have not been run yet.

## At a glance

```
Stages 1-2     run_local_lqr.py                local LQR gain K, test runs, maps (main + model error)
                   |
Stage 3        run_koopman.py --stage data     train / val / test trajectories -> ident_data.npz
                   |
Stage 4        run_koopman.py --stage fit      one lifted model per observable set -> koopman_*.npz
                   |
Stage 5        run_koopman.py --stage design   LQR per model + true-plant stability check
                   |
Stage 5        run_koopman.py --stage maps     Koopman maps, same conditions, grid and seeds
                   |
Stage 5        run_koopman.py --stage figures  maps, who-wins, by-distance, time series, tables
                   |
Exp. 3         run_tracking.py                 small scene (C near upright) and large scene
                                               (C, G, O around the shoulder), with GIFs
                   |
Stage 6        Report and presentation         not started

Optional       run_koopman.py --stage extra    design scan, Lyapunov certificates, convergence rates

Shared modules (used by every script):  config.py  plant.py  tool_lib.py  koopman.py  control.py
Hardware profile only:                  hardware.py
```

Run the scripts top to bottom: `run_koopman.py` and `run_tracking.py` need files written by the scripts above them, and all three import the shared modules. The import order of the modules is one-way: `config` → `plant` → `tool_lib` → `koopman` → `control`.

## Where each stage stands

The main balance comparison is complete for the ideal plant (`PLANT_PROFILE = "ideal"`, `REALISM = "ideal"`); the new conditions and the tracking experiment still need their first run.

| Stage | Run with | Status | Main outputs | Headline result |
| --- | --- | --- | --- | --- |
| 1. Define and verify the plant | `python plant.py` | Done | `figures/conventions.png` | Target x\* = [π, 0, 0, 0] (upright), dt = 5 ms, torque limit [4, 1] N·m; fast copy matches DFKI to ~1e-13 |
| 2. Local-linearization LQR | `run_local_lqr.py` | Done | `results/local_lqr_*.npz`, local-LQR maps | Converges from 90.8% of the initial-angle slice; local rate α = 3.22 s⁻¹ |
| 3. Identification data | `run_koopman.py --stage data` | Done | `results/ident_data.npz` | 155,195 / 55,146 / 52,678 train / validation / test pairs, split by trajectory |
| 4. Koopman predictors | `run_koopman.py --stage fit` | Done | `results/koopman_*.npz`, `koopman_prediction.png` | Trig (8 observables) has the lowest 0.25 s error inside the training region: 0.022 rad vs 0.056 rad for Taylor |
| 5. Balance comparison | `run_koopman.py --stage design, maps, figures` | Main condition done; distance figure and model error not yet run | maps, `by_distance.png`, `robustness.png`, summary tables | Trig Koopman LQR converges from 98.5% vs 90.8%; cubic and 13-term trig gains destabilize the real plant |
| Exp. 3: tracking letters | `run_tracking.py` | Both scenes run once in earlier versions; the two-scene version with model error not yet run | `tracking_*_<scene>_<letter>.png`, `tracking_<scene>_<letter>.gif`, `summary_scenes.md` | Small C near upright: Koopman best. Letters around the shoulder: physics LQR best |
| 6. Report and presentation | — | Not started | — | — |

The hardware profile (`PLANT_PROFILE = "hardware"`) has only the stage-2 baseline so far: 100% success with ideal sensing, 50% with realistic sensing.

## What the results say so far

Near the target, every Koopman controller is effectively linear: ψ(e) ≈ J e, so u ≈ u\* − (K_z J) e, and K_z J came out almost the same as the local LQR gain.

| Region | Prediction (the model) | Balance (control) | Tracking |
| --- | --- | --- | --- |
| Small (near upright) | Koopman better: trig error 0.022 rad vs 0.056 rad for Taylor over 0.25 s | Tie: same local rate (3.24 vs 3.22 s⁻¹), costs within 1% | Koopman better (small C) |
| Edge (motors saturating) | — | Trig Koopman better: 98.5% vs 90.8% success comes from starts here | — |
| Large (whole workspace) | Koopman worse: one linear model cannot cover it | Polynomial Koopman worse or unstable | Physics LQR better (letters around the shoulder) |
| Exact physics model available | — | Gravity-compensated LQR beats every Koopman controller (99.4%) | Physics LQR wins with an exact model |

The defensible claim: inside its training region, Koopman predicts more accurately. That shows up as better control when the arm moves through the region (tracking) or pushes against the torque limits (the edge of the balance map). It makes no difference for simply holding the arm near the target. The two runs still to do test this directly: `by_distance.png` groups the balance maps by distance from upright, and model error asks whether data beats a wrong physics model.

Physics knowledge also enters the Koopman pipeline in four places: the closed-loop identification runs are stabilized by local LQR, the suitability check uses the true plant's Jacobian, the trig observables come from the gravity terms, and u\* comes from the physics model (it is 0 at upright). The model fit and the gain design themselves use data only.

## The variables

Every value you change lives in section 1 of `config.py`; every script reads it. Robustness variables change one at a time; the others stay at their main value.

| Variable | Balance (Exp. 1, 2) | Tracking (Exp. 3) | Values | Config name |
| --- | --- | --- | --- | --- |
| Initial angles | Main: a grid inside every condition | Fixed: starts on the letter | 31 × 31 over ±π | `MAP_N`, `MAP_RANGE` |
| Region size | Distance bands of the map | Two scenes: small and large | bands 0–0.5, 0.5–1, 1–1.5, 1.5–2, 2–3, 3–4.5 rad; small = 10 cm C at 0.4 m above the shoulder, 2 s, [4, 1] N·m; large = 60 cm C, G, O around the shoulder, 4 s, [4, 2] N·m | `DISTANCE_EDGES`, `TRACK_SCENES`, `TRACK_RUN` |
| Observables | Main: one Koopman controller per set | Main: one Koopman controller per set | state only (4), quadratic (14), cubic (34), trig (8), trig (13) | `OBSERVABLES` |
| Model error (physics designs only) | Robustness (default) | Robustness (default) | design masses ×0.8, ×1.2 | `MODEL_ERROR_SWEEP`, `ROBUSTNESS`, `TRACK_ROBUSTNESS` |
| Torque limit, initial velocity, plant change, noise | Optional robustness | Optional (plant, noise) | as before | add `"torque"`, `"velocity"`, `"plant"`, `"noise"` to `ROBUSTNESS` / `TRACK_ROBUSTNESS` |

Model error is the default robustness test because it asks the research question: the physics-based designs get wrong masses while Koopman learns from the true arm. Plant change and noise hurt every controller alike, so they are optional. In tracking, the physics controllers are designed under each condition (so model error reaches them), and the Koopman controllers are fitted once per letter from the true arm's data.

## Run it locally

Run these in order from the repo root with `conda activate ece627`; each step only needs the files the previous one wrote. First delete the 15 old files the consolidation replaced: `plant_utils.py`, `dynamics_np.py`, `experiments.py`, `metrics.py`, `sweep.py`, `fast_sweep.py`, `viz.py`, `style.py`, `reference.py`, `collect_data.py`, `lqr_design.py`, `controllers.py`, `stability.py`, `tracking.py`, `scan_designs.py`.

1. `python plant.py` (a few seconds). Check that `max |rhs_vec - rhs_DFKI|` is around 1e-13 and the inverse-dynamics check is near zero. If the first is large, your DFKI version differs from commit `1e8544e`.
2. `python run_local_lqr.py`. Check that all five test cases print `yes` and that the time-series figures and the `.gif` appear in `figures/`.
3. `python run_local_lqr.py --maps all`. Local-LQR maps for the main condition and model error; the main map should show about 91% success. Use `--maps main` first if you want a quick check.
4. `python run_koopman.py`. The core stages in order, or one at a time:
   1. `--stage data`: identification data; check the pair counts match the table above.
   2. `--stage fit`: fits every observable set, writes `koopman_prediction.png`.
   3. `--stage design`: LQR per model, prints `suitable=True/False`, writes `koopman_eigenvalues.png`.
   4. `--stage maps` (add `--maps main` to skip robustness): Koopman maps for every condition.
   5. `--stage figures`: success maps, who-wins, `by_distance.png`, time series, summary tables, `robustness.png`.
5. `python run_tracking.py --check`, then `python run_tracking.py`. The check confirms each letter is reachable and within its scene's torque limit. `--scenes small` runs one scene, `--figures` redraws from saved results and `--no-gif` skips the animations.
6. Optional: `python run_koopman.py --stage extra` for the design scan, Lyapunov certificates and convergence rates. The comparison does not need them.

Noisy conditions always use DFKI's simulator (the fast one has no noise), so they are the slowest maps. Before committing, add `figures/`, `results/` and `__pycache__/` to `.gitignore`; every file in them is regenerated.

## Shared modules

Five modules are imported by every script; together they guarantee that every controller sees the same plant, cost, saturation, conditions and metrics. Every balance controller in the project, local or Koopman, is the same law with a different lift ψ:

```
u = sat( u* - K psi(x - x*) )
```

Tracking uses the same idea with a moving target: u = sat(u_ff(t) − K(t) e(t)).

| File | Role | Key functions | Read it to understand |
| --- | --- | --- | --- |
| `config.py` | Every number in one place. Section 1 is the experiment variables; 2 fixed design choices; 3 plant; 4 Koopman data and fitting; 5 stability analysis | constants only | What is varied, what is held fixed, and the switches `PLANT_PROFILE` and `REALISM` |
| `plant.py` | DFKI plant and simulator (true arm or design model), equilibrium, linearization, zero-order hold, and a vectorized copy of DFKI's equations with inverse dynamics | `make_plant`, `make_simulator`, `linearize`, `c2d_zoh`, `VecPlant`, `check_against_dfki` | Angle wrapping; why the design model drops Coulomb friction; how model error and plant change scale the masses |
| `tool_lib.py` | Conditions, simulation, metrics, map sweeps, tracking references, figure style and all plotting | `Condition`, `conditions`, `applied`, `simulate`, `run_map`, `evaluate`, `build_reference`, `tracking_metrics`, `plot_map`, `title` | How one config change becomes a named condition shared by every script; the success test, settling time, RMSE, cost J, effort, saturation, rate α̂ |
| `koopman.py` | Identification data, observables, ridge least-squares fit, multistep prediction | `generate`, `load_data`, `pairs`, `make_lift`, `fit`, `fit_dictionary`, `multistep_rmse` | How ψ is built; why training is restricted to the region near upright |
| `control.py` | Every controller design: LQR with PBH checks, local LQR, Koopman LQR and its stability checks, tracking controllers | `dlqr`, `design_local_lqr`, `LiftedLQRController`, `design`, `true_local_eigs`, `roa_certificate`, `TrackingController`, `physics_controllers`, `koopman_controllers` | The two control laws and how a Koopman gain is checked against the real plant |

Two conventions trip people up. Angles follow DFKI: q1 = 0 hangs down and q2 is relative to link 1, so upright is q1 = π. DFKI writes gravity as −G, so its G is −g(q) in your course notation.

## The three scripts

Each script reads its settings from `config.py`, writes to `results/` and `figures/`, and puts the varied and fixed settings in every figure title.

### `run_local_lqr.py` — Simulation 1, stages 1–2

Calls `design_local_lqr()` from `control.py` (solves for u\*, linearizes at upright, discretizes with zero-order hold, calls `dlqr`), then runs five starts from 0.05 to 2.0 rad, saves time-series plots and a GIF. `--maps main|robust|all` sweeps the initial-angle grid under each condition; `--live` opens DFKI's animation. The baseline gain is defined only in `control.py`, so every script uses the same one.

### `run_koopman.py` — Simulation 2, stages 3–5

Five core stages that pass results as files, so you can rerun one:

- `data`: 600 closed-loop runs (local LQR plus random torque, 1.5 s) and 1,200 short open-loop bursts (0.4 s), cut where an angle wraps or a velocity exceeds 15 rad/s, split by whole trajectory. Short bursts matter: long ones spend most of their time swinging near hanging-down and pull the fit away from upright.
- `fit`: one model per observable set, trained on |Δq| ≤ 0.6 rad, ridge weight chosen by 0.25 s multistep error on validation windows.
- `design`: LQR on each lifted model (Q_z = CᵀQC, same R). A model is marked suitable only if its gain stabilizes the true plant locally, tested with K · ∂ψ/∂x(0) against the real Jacobian (`true_local_eigs`), not the fitted model.
- `maps`: Koopman maps for every condition, on the same grid and seeds as local LQR.
- `figures`: success maps, who-wins (ties within 1% marked), success and cost by distance from upright, time series, robustness panels, summary tables.

`--stage extra` holds the optional analysis: the design scan (training radius × observables × gain design on 400 validation starts, which is how r = 0.6 rad and infinite-horizon LQR were chosen), sampled Lyapunov certificates, and convergence-rate figures.

### `run_tracking.py` — Experiment 3

Two scenes ask the same small-vs-large question as the balance maps. The small scene draws a 10 cm C at 0.4 m above the shoulder in 2 s with the balance torque limit [4, 1] N·m, staying near upright. The large scene draws 60 cm C, G and O centred on the shoulder in 4 s each, so the arm sweeps the whole workspace. Its torque limit is [4, 2] N·m, because the elbow needs about 1 N·m just to hold link 2 level. Paths use minimum-jerk timing and elbow-up inverse kinematics. Where the large letters pass hanging-down, the controllers use the shortest-angle error so it never jumps by 2π.

Controllers: `local_fixed` (upright gain plus inverse-dynamics feedforward), `tvlqr` (time-varying LQR along the path) and one Koopman controller per observable set, refit on data covering the letter, with feedforward solved from the lifted model. Each run scores tip error, shape error (mean distance to the letter), effort and saturation. `summary_scenes.md` puts every scene, letter and condition side by side with the best controller, and `tracking_<scene>_<letter>.gif` shows every controller's arm drawing the letter.

### `hardware.py` — future rig

Turns parts (GM5208-12 shoulder, GM3506 elbow, 3D-printed 12 / 10 cm links, 15 g tip mass) into DFKI plant parameters with `build()`. `config.py` uses it only when `PLANT_PROFILE = "hardware"`. All numbers are datasheet values or estimates until the rig is identified.

## Outputs: which file answers which question

The figures map onto the four figures your project plan asked for, plus the ones that explain the mechanisms and the two extensions. `tool_lib.py` fixes one colour, line style and marker per controller, so a controller looks the same in every figure.

| Question | Figure or table | Made by |
| --- | --- | --- |
| What do joint angles and torques do over time? | `timeseries_*.png`, `local_lqr_*_overlay.png` | `run_koopman.py --stage figures`, `run_local_lqr.py` |
| How does multistep prediction error change with observable count? | `koopman_prediction.png` | `run_koopman.py --stage fit` |
| Where does each controller converge or fail? | `maps_success_all.png`, `map_local__*.png` | `run_koopman.py --stage figures`, `run_local_lqr.py --maps` |
| Where is Koopman better or worse than local LQR? | `maps_who_wins.png` (ties within 1% marked) | `run_koopman.py --stage figures` |
| Is Koopman better close to the target or far away? | `by_distance.png`, `results/summary_distance_*.md` | same |
| Does data beat a wrong physics model? | `robustness.png`, `results/summary_robustness_*.md` | same |
| Summary of control performance | `results/summary_*.md` and `.csv`, `results/summary_slide_*.md` | same |
| Does the model's stability guarantee hold on the real plant? | `koopman_eigenvalues.png` | `run_koopman.py --stage design` |
| Which controller follows a path best, small vs large? | `tracking_paths_<scene>_<letter>.png`, `tracking_timeseries_<scene>_<letter>.png`, `tracking_<scene>_<letter>.gif`, `results/tracking/*/summary_scenes.md` | `run_tracking.py` |
| Optional: certified regions, convergence rates, design sensitivity | `roa_shapes.png`, `convergence_rates.png`, `scan_sensitivity.png` | `run_koopman.py --stage extra` |

## Known gaps and next steps

The simplified code passes static checks but has not been run yet.

- [ ] Run the core pipeline once (`plant.py`, `run_local_lqr.py --maps all`, `run_koopman.py`) and check the main numbers match the table above.
- [ ] Read `by_distance.png`: does it show a tie near upright, Koopman ahead at the edge, and Koopman behind far away?
- [ ] Read the model-error results (balance and tracking): does Koopman catch up with physics LQR when the physics model is wrong?
- [ ] Run Experiment 3: `run_tracking.py --check` first, then the full run; compare the scenes in `summary_scenes.md`.
- [ ] In the report, call the trig observables physics-informed (gravity is linear in sin Δq), and say which steps use physics knowledge (see "What the results say so far").
- [ ] Optional: a gravity-compensated LQR baseline (scratch checks: 99.4% in the ideal case), a strict data-only Koopman run, the hardware profile.
- [ ] Write the report and presentation (stage 6).

## A reading path

Read the files in this order; each builds on the last, and each exercise takes a few minutes and has a known answer to check against.

1. `config.py`, section 1. Set `TORQUE_LIMIT_MAIN = [8.0, 2.0]` and `MAP_N = 15`, run `python run_local_lqr.py --maps main`, and expect about 100% success; revert. This is why the torque limit decides whether there is anything to compare.
2. `plant.py`. In Python, run `from control import design_local_lqr`, then `d = design_local_lqr()` and print `np.linalg.eigvals(d["Ac"])`; expect two positive eigenvalues (about +7.6 and +4.7 s⁻¹), the unstable upright modes.
3. `control.py`, LQR part (`dlqr`, `design_local_lqr`, `LiftedLQRController`). Double `R` in `config.py`, rerun `python run_local_lqr.py`, and watch settling times grow. Revert.
4. `tool_lib.py`, conditions and `evaluate`. Run one baseline case and print `evaluate(T, X, U, u_star)["alpha_hat"]`; it should be near the theoretical 3.22 s⁻¹. Then read `conditions()` to see how section 1 becomes the list of runs.
5. `koopman.py`, data part. Load `kp.load_data()` and histogram the Δq₁ values of the training pairs; about 41% lie inside ±0.6 rad.
6. `koopman.py`, fitting part. Fit the state-only model on all data and on |Δq| ≤ 0.6 rad (snippet below) and compare its largest eigenvalue magnitude (1.013 and 1.032) with the Taylor model's 1.039. The all-data fit comes out smaller: that is secant bias.
7. `control.py`, stability part. For the cubic model, print `rate(true_local_eigs(Ad, Bd, K, lift))`; it is strongly negative, which is why `--stage design` marks it unsuitable.
8. `run_koopman.py`, then `run_tracking.py`. Run `python run_koopman.py --stage design` and match its `suitable=` lines to the summary table; then read how `run_tracking.py` uses the same observables with a moving target in its two scenes.

```python
import numpy as np, koopman as kp
Xk, Uk, Xk1 = kp.pairs(kp.load_data()["train"])
for r in (np.pi, 0.6):
    m = kp.region_mask(Xk, r) & kp.region_mask(Xk1, r)
    lift = kp.make_lift("poly1", np.ones(4))
    A, B = kp.fit(lift, Xk[m], Uk[m], Xk1[m])
    print(r, np.abs(np.linalg.eigvals(A)).max())
```
