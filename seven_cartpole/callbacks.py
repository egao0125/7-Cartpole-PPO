from __future__ import annotations

import pathlib

import imageio.v3 as iio
from stable_baselines3.common.callbacks import BaseCallback

from seven_cartpole.env import EnvConfig, SevenPendulumCartpoleEnv
from seven_cartpole.render import draw_hud


class TrainingVideoCallback(BaseCallback):
    def __init__(
        self,
        *,
        video_dir: pathlib.Path,
        video_freq: int,
        max_episode_steps: int,
        init_angle_noise: float,
        fps: int = 50,
        width: int = 1280,
        height: int = 720,
        seed: int = 1000,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.video_freq = video_freq
        self.max_episode_steps = max_episode_steps
        self.init_angle_noise = init_angle_noise
        self.fps = fps
        self.width = width
        self.height = height
        self.seed = seed
        self._next_video_at = video_freq

    def _on_training_start(self) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.video_freq <= 0:
            return True
        if self.num_timesteps < self._next_video_at:
            return True

        self._record_video()
        while self._next_video_at <= self.num_timesteps:
            self._next_video_at += self.video_freq
        return True

    def _record_video(self) -> None:
        config = EnvConfig(
            max_episode_steps=self.max_episode_steps,
            init_angle_noise=self.init_angle_noise,
        )
        env = SevenPendulumCartpoleEnv(
            config=config,
            render_mode="rgb_array",
            width=self.width,
            height=self.height,
        )
        obs, info = env.reset(seed=self.seed + self.num_timesteps)
        frames = []
        episode_return = 0.0
        healthy_frames = 0

        try:
            for step in range(self.max_episode_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                healthy_frames += 1 if info["is_healthy"] else 0

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
                        "policy_name": f"train_step_{self.num_timesteps}",
                    },
                )
                frames.append(frame)

                if terminated or truncated:
                    break
        finally:
            env.close()

        if frames:
            output = self.video_dir / f"step_{self.num_timesteps:09d}.mp4"
            iio.imwrite(output, frames, fps=self.fps)
            print(f"Wrote training video {output}")
