#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import platform
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")

import imageio.v3 as iio
import numpy as np
from stable_baselines3 import PPO

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seven_cartpole import SevenPendulumCartpoleEnv
from seven_cartpole.env import EnvConfig
from seven_cartpole.policies import heuristic_action_from_obs
from seven_cartpole.render import draw_hud


def evaluate(args):
    config = EnvConfig(max_episode_steps=args.max_episode_steps, init_angle_noise=args.init_angle_noise)
    env = SevenPendulumCartpoleEnv(
        config=config,
        render_mode="rgb_array" if args.output else None,
        width=args.width,
        height=args.height,
    )
    model = PPO.load(args.model, device=args.device) if args.model else None
    policy_name = pathlib.Path(args.model).stem if args.model else "heuristic"

    episode_returns = []
    episode_lengths = []
    episode_successes = []
    frames = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        episode_return = 0.0
        healthy_frames = 0

        for step in range(args.max_episode_steps):
            if model is None:
                action = heuristic_action_from_obs(obs)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            healthy_frames = healthy_frames + 1 if info["is_healthy"] else 0

            if args.output and episode == 0 and step % args.render_every == 0:
                frame = env.render()
                frame = draw_hud(
                    frame,
                    {
                        "step": step,
                        "sim_time": info["sim_time"],
                        "action": info["action"],
                        "episode_return": episode_return,
                        "reward": float(reward),
                        "tip_height": info["tip_height"],
                        "cart_x": info["cart_x"],
                        "uprightness": info["uprightness"],
                        "is_healthy": info["is_healthy"],
                        "healthy_frames": healthy_frames,
                        "policy_name": policy_name,
                    },
                )
                frames.append(frame)

            if terminated or truncated:
                episode_lengths.append(step + 1)
                episode_returns.append(episode_return)
                episode_successes.append(bool(info["success"]))
                break

    env.close()

    print(f"episodes: {len(episode_returns)}")
    print(f"mean_return: {np.mean(episode_returns):.3f}")
    print(f"mean_length: {np.mean(episode_lengths):.1f}")
    print(f"success_rate: {np.mean(episode_successes):.3f}")

    if args.output:
        output = ROOT / args.output
        iio.imwrite(output, frames, fps=args.fps)
        print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to a Stable-Baselines3 PPO zip. Omit for heuristic.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--init-angle-noise", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
