#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import platform
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seven_cartpole import SevenPendulumCartpoleEnv
from seven_cartpole.callbacks import TrainingVideoCallback
from seven_cartpole.env import EnvConfig


def sync_to_gcs(run_dir: pathlib.Path, gcs_path: str | None, run_name: str) -> None:
    if not gcs_path:
        return
    destination = f"{gcs_path.rstrip('/')}/{run_name}"
    subprocess.run(
        ["gcloud", "storage", "rsync", "-r", str(run_dir), destination],
        check=True,
    )
    print(f"Synced run outputs to {destination}")


def make_env(max_episode_steps: int, init_angle_noise: float):
    def _factory():
        config = EnvConfig(max_episode_steps=max_episode_steps, init_angle_noise=init_angle_noise)
        return Monitor(SevenPendulumCartpoleEnv(config=config))

    return _factory


def train(args):
    run_dir = ROOT / "runs" / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best"
    log_dir = run_dir / "tensorboard"
    video_dir = run_dir / "videos"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        make_env(args.max_episode_steps, args.init_angle_noise),
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
        seed=args.seed,
    )
    eval_env = Monitor(
        SevenPendulumCartpoleEnv(
            config=EnvConfig(
                max_episode_steps=args.max_episode_steps,
                init_angle_noise=args.init_angle_noise,
            )
        )
    )

    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {"pi": [256, 256, 256], "vf": [256, 256, 256]},
    }

    if args.load:
        model = PPO.load(args.load, env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(log_dir),
            device=args.device,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_7cartpole",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        ),
    ]
    if args.video_freq > 0:
        callbacks.append(
            TrainingVideoCallback(
                video_dir=video_dir,
                video_freq=args.video_freq,
                max_episode_steps=args.video_max_episode_steps,
                init_angle_noise=args.init_angle_noise,
                fps=args.video_fps,
                width=args.video_width,
                height=args.video_height,
                seed=args.seed + 10_000,
            )
        )

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        tb_log_name=args.run_name,
        reset_num_timesteps=not bool(args.load),
    )
    model.save(run_dir / "final_model")
    sync_to_gcs(run_dir, args.gcs_path, args.run_name)
    env.close()
    eval_env.close()
    print(f"Wrote run outputs to {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="ppo_7link")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--load", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--init-angle-noise", type=float, default=0.025)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--video-freq", type=int, default=100_000)
    parser.add_argument("--video-max-episode-steps", type=int, default=500)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--gcs-path", default=None, help="Optional gs:// bucket prefix to sync run outputs after training.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
