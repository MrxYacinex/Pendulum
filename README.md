# Pendulum Control Sandbox

This repository contains experiments for stabilizing a triple-pendulum-on-cart system.

The main training and simulation entrypoint is:

- `triple_pendulum_rl.py`

It includes:

- nonlinear triple-pendulum dynamics
- force-driven cart actuation model
- PPO training (single or parallel envs)
- fixed-seed evaluation metrics
- live visualization modes
- LQR sanity diagnostics and feasibility sweep modes

## Project State

There are multiple experiment files and generated artifacts in this repo. The actively maintained RL pipeline is in `triple_pendulum_rl.py`.

Other folders/files (like `simulation.py`, `sim_looking.tsx`, `triple-pendulum/`) are preserved as related experiments/prototypes.

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install numpy scipy matplotlib gymnasium stable-baselines3 torch
```

## Quick Start

### 1) Fast training

```bash
python triple_pendulum_rl.py --mode train_fast --steps 600000 --num-envs 8 --model ppo_triple_pendulum_upright_fast
```

### 2) Resume training

```bash
python triple_pendulum_rl.py --mode train --steps 1000000 --num-envs 8 --model ppo_triple_pendulum_upright_fast --load-model ppo_triple_pendulum_upright_fast --eval-every 50000 --eval-seeds 8 --eval-horizon 1200
```

### 3) Visualize a trained model

```bash
python triple_pendulum_rl.py --mode viz --model ppo_triple_pendulum_upright_fast
```

### 4) Plot training curve

```bash
python triple_pendulum_rl.py --mode curve --logdir rl_logs
```

## Training Modes

- `train`: training with fixed-eval callback and logging
- `train_fast`: speed-first training (reduced overhead)
- `train_live`: training + live plotting/animation
- `train_debug_easy`: simplified learnability-check configuration
- `train_viz`: train then visualize
- `viz`: load model and rollout animation
- `curve`: reward curve from monitor logs
- `sanity_lqr`: local controllability/stabilization diagnostic
- `sanity_sweep`: sweep LQR feasibility vs input/bound limits

## Metrics

Fixed eval writes:

- `rl_logs/fixed_eval_metrics.csv`

Columns:

- `timesteps`
- `mean_return`
- `success_rate`
- `mean_balanced_fraction`
- `mean_sat_fraction`
- `mean_max_cart`
- `mean_abs_u`

Tip: compare only rows from the same run block/logdir.

## Recommended Workflow

1. Run `sanity_lqr` / `sanity_sweep` to verify local feasibility.
2. Train with `train_debug_easy` until success starts moving off zero.
3. Resume into `train` for realistic conditions.
4. Validate with `viz` and fixed-eval metrics.

## Windows OpenMP Note

If you hit:

`OMP: Error #15: ... libiomp5md.dll already initialized`

run commands in PowerShell as:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; python triple_pendulum_rl.py --mode sanity_sweep --sanity-episodes 20 --eval-horizon 1200
```

This is a local runtime conflict workaround.

## Generated Artifacts

Training produces:

- model weights (`*.zip`)
- vecnormalize stats (`*.vecnormalize.pkl`)
- model metadata (`*.meta.json`)
- logs (`rl_logs/`)

These should generally not be committed unless you intentionally want to version checkpoints.

