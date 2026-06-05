# 7-Link MuJoCo Cart-Pole PPO

This repo trains and evaluates a 7-link MuJoCo cart-pole controller with PPO.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

For headless cloud runs:

```bash
export MUJOCO_GL=egl
```

## Render a Video

```bash
python3 run_seven_cartpole.py --mode video --duration 8
```

The script writes `seven_pendulum_cartpole.mp4`.

## PPO Training

The environment uses a 27-value observation:

```text
cart_x, cart_v
sin(theta_1..7), cos(theta_1..7)
theta_dot_1..7
tip_x, tip_z, tip_vx, tip_vz
```

The PPO actor and critic are separate 3-layer MLPs:

```text
Actor:  27 -> 256 -> 256 -> 256 -> 1
Critic: 27 -> 256 -> 256 -> 256 -> 1
```

Start a short smoke training run:

```bash
python3 scripts/train_ppo.py --total-steps 4096 --n-envs 1 --run-name smoke
```

Start a real run:

```bash
python3 scripts/train_ppo.py \
  --total-steps 1000000 \
  --n-envs 8 \
  --run-name ppo_7link
```

Outputs are written under `runs/<run-name>/`.

## Evaluate

Evaluate the heuristic baseline:

```bash
python3 scripts/eval_policy.py --episodes 5
```

Render a PPO checkpoint:

```bash
python3 scripts/eval_policy.py \
  --model runs/ppo_7link/best/best_model.zip \
  --episodes 5 \
  --output ppo_7link_eval.mp4
```

## Live Viewer

On macOS, MuJoCo's passive viewer should be launched with `mjpython`:

```bash
mjpython run_seven_cartpole.py --mode viewer
```

Controls:

- `r`: reset near upright
- `m`: toggle manual control
- `a`: push left in manual mode
- `d`: push right in manual mode
- `s`: stop pushing in manual mode

The included `heuristic_policy` is only a crude baseline. Seven links is a hard control problem; the intended next step is to replace that function with your trained RL policy.
