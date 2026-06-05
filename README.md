# 7-Link MuJoCo Cart-Pole PPO Swing-Up

This repo trains and evaluates a 7-link MuJoCo cart-pole swing-up controller with PPO.

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

The default task is now `swingup`:

```text
start: first link hanging down
goal: swing the full chain upward
success: hold the upright health condition for 100 consecutive RL steps
failure: cart leaves bounds or simulation state becomes invalid
```

The old near-upright balancing setup is still available with:

```bash
python3 scripts/train_ppo.py --task balance ...
```

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

Training writes three visual/debug streams:

```text
runs/<run-name>/tensorboard/   scalar graphs
runs/<run-name>/videos/        periodic MP4 policy previews
runs/<run-name>/eval/          evaluation metrics
```

Open training graphs:

```bash
tensorboard --logdir runs/ppo_7link/tensorboard --port 6006
```

Record preview videos during training:

```bash
python3 scripts/train_ppo.py \
  --total-steps 1000000 \
  --n-envs 8 \
  --run-name ppo_7link \
  --video-freq 50000 \
  --video-seconds 20
```

Set `--video-freq 0` to disable periodic videos.

To upload each training video to a GitHub Release, set `GITHUB_TOKEN` in the training environment and pass `--github-repo`:

```bash
export GITHUB_TOKEN=github_pat_or_classic_token_with_repo_access

python3 scripts/train_ppo.py \
  --run-name ppo_7link_runpod \
  --total-steps 1000000 \
  --n-envs 16 \
  --device cuda \
  --video-freq 50000 \
  --video-seconds 20 \
  --github-repo egao0125/7-Cartpole-PPO \
  --github-release-tag ppo-7link-training-videos
```

Videos are uploaded as assets on the `ppo-7link-training-videos` GitHub Release. This avoids committing MP4 files into the repository history.

## Health, Failure, And Success

The environment uses one shared health definition for reward, termination, and evaluation:

```text
healthy =
  all MuJoCo state values are finite
  and abs(cart_x) <= 2.15
  and tip_height >= 3.0
  and mean(cos(theta_1..theta_7)) >= 0.65
```

Failure:

```text
cart out of bounds or non-finite state: terminate immediately
swingup: tip/posture unhealthy is allowed so the policy can recover
swingup: low, nearly motionless dead-hang state resets after a warmup window
balance: tip/posture unhealthy terminates after the warmup period
```

Success:

```text
swingup: healthy for 100 consecutive RL steps
balance: episode reaches max_episode_steps while still healthy
```

Reward uses the same health condition for the alive bonus, plus dense shaping:

```text
balance reward =
  2.0   * tip_height
  + 0.5   * mean_link_uprightness
  - 0.3   * abs(cart_x)
  - 0.02  * cart_velocity^2
  - 0.01  * action^2
  - 0.002 * sum(theta_dot_i^2)
  + alive_bonus_if_healthy
```

Swing-up uses a different reward model. It is designed to avoid the "do nothing while hanging" local optimum:

```text
swingup reward =
  normalized_tip_height
  + global_link_uprightness
  + positive_tip_height_progress
  + upward_tip_velocity
  + cart_motion_reward_while_low
  + action_pump_reward_while_low
  + healthy_hold_bonus_once_upright
  - cart_position_penalty_only_when_high
  - stabilization_action/velocity_penalties_only_when_high
```

The key distinction is that swing-up does not punish being down early. It rewards pumping energy into the system and only becomes conservative once the chain is near upright. The cart still has a hard track limit; crossing that bound resets the episode. A failed attempt that falls back to a low, nearly static dead-hang also resets so training does not spend long rollouts doing nothing.

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

## GPU And Cloud Storage

This should run on Google Cloud, not on a local Mac, for serious training.

From Google Cloud Shell, create a GPU VM and start training:

```bash
git clone https://github.com/egao0125/7-Cartpole-PPO.git
cd 7-Cartpole-PPO

PROJECT=your-gcp-project-id \
BUCKET=your-unique-cartpole-bucket-name \
ZONE=us-central1-a \
VM_NAME=seven-cartpole-gpu \
MACHINE_TYPE=g2-standard-16 \
GPU_TYPE=nvidia-l4 \
GPU_COUNT=1 \
IMAGE_FAMILY=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
RUN_NAME=ppo_7link_gpu \
TOTAL_STEPS=1000000 \
N_ENVS=16 \
bash cloud/create_gcp_gpu_vm.sh
```

The VM startup script clones this repo, installs dependencies, verifies CUDA, starts PPO, writes TensorBoard logs/videos/checkpoints, and syncs run outputs to Cloud Storage.

To watch logs from Cloud Shell:

```bash
gcloud compute ssh seven-cartpole-gpu \
  --zone=us-central1-a \
  --command='tail -f /opt/7-Cartpole-PPO/runs/ppo_7link_gpu/train.log'
```

To list videos/checkpoints in Cloud Storage:

```bash
gcloud storage ls gs://your-unique-cartpole-bucket-name/seven-cartpole/runs/ppo_7link_gpu/**
```

On any CUDA machine, Stable-Baselines3/PyTorch can use GPU:

```bash
python3 scripts/train_ppo.py \
  --run-name ppo_7link_gpu \
  --total-steps 1000000 \
  --n-envs 16 \
  --device cuda
```

Check the VM sees the GPU:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

For Google Cloud, use a PyTorch Deep Learning VM with an NVIDIA L4 or better. MuJoCo rendering should run headless:

```bash
export MUJOCO_GL=egl
```

Create a durable Cloud Storage bucket for checkpoints and videos:

```bash
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-central1
```

Sync a run to Cloud Storage after training:

```bash
python3 scripts/train_ppo.py \
  --run-name ppo_7link_gpu \
  --total-steps 1000000 \
  --n-envs 16 \
  --device cuda \
  --gcs-path gs://YOUR_BUCKET_NAME/seven-cartpole/runs
```

That writes local outputs under `runs/ppo_7link_gpu/` and then syncs them to:

```text
gs://YOUR_BUCKET_NAME/seven-cartpole/runs/ppo_7link_gpu/
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
