#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/egao0125/7-Cartpole-PPO.git}"
WORKDIR="${WORKDIR:-/opt/7-Cartpole-PPO}"
RUN_NAME="${RUN_NAME:-ppo_7link_gpu}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
N_ENVS="${N_ENVS:-16}"
VIDEO_FREQ="${VIDEO_FREQ:-50000}"
VIDEO_MAX_EPISODE_STEPS="${VIDEO_MAX_EPISODE_STEPS:-500}"
GCS_PATH="${GCS_PATH:-}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

sudo apt-get update
sudo apt-get install -y git python3-venv ffmpeg

if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" pull --ff-only
else
  sudo mkdir -p "$(dirname "$WORKDIR")"
  sudo chown "$USER:$USER" "$(dirname "$WORKDIR")"
  git clone "$REPO_URL" "$WORKDIR"
fi

cd "$WORKDIR"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

nvidia-smi || true
.venv/bin/python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

mkdir -p "runs/${RUN_NAME}"

GCS_ARGS=()
if [ -n "$GCS_PATH" ]; then
  GCS_ARGS=(--gcs-path "$GCS_PATH")
fi

.venv/bin/python -u scripts/train_ppo.py \
  --run-name "$RUN_NAME" \
  --total-steps "$TOTAL_STEPS" \
  --n-envs "$N_ENVS" \
  --device cuda \
  --video-freq "$VIDEO_FREQ" \
  --video-max-episode-steps "$VIDEO_MAX_EPISODE_STEPS" \
  --checkpoint-freq 100000 \
  --eval-freq 50000 \
  --eval-episodes 5 \
  "${GCS_ARGS[@]}" \
  2>&1 | tee -a "runs/${RUN_NAME}/train.log"
