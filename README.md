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
- [ ] Verify the sample simulation and animation.
- [ ] Define plant parameters and target equilibrium.
- [ ] Implement the local-linearization LQR baseline.
- [ ] Collect trajectories and identify a Koopman model.
- [ ] Compare closed-loop performance.

## Running

Run the initial simulation example:

    python sample.py

## Experiments and Results

To be added: experiment settings, prediction errors,
control performance, and limitations.

## Reproducibility

DFKI commit: to be recorded.

A verified environment specification will be added later.