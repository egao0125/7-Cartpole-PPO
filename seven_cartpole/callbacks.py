from __future__ import annotations

import pathlib
import urllib.error
import urllib.parse
import urllib.request

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
        task: str,
        init_angle_noise: float,
        fps: int = 50,
        video_seconds: float = 20.0,
        width: int = 1280,
        height: int = 720,
        seed: int = 1000,
        github_repo: str | None = None,
        github_token: str | None = None,
        github_release_tag: str | None = None,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.video_freq = video_freq
        self.max_episode_steps = max_episode_steps
        self.task = task
        self.init_angle_noise = init_angle_noise
        self.fps = fps
        self.video_seconds = video_seconds
        self.width = width
        self.height = height
        self.seed = seed
        self.github_repo = github_repo
        self.github_token = github_token
        self.github_release_tag = github_release_tag
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
            task=self.task,
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
        episode = 0
        target_frames = max(1, int(round(self.video_seconds * self.fps)))

        try:
            step = 0
            while len(frames) < target_frames:
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
                        "healthy_streak": info["healthy_streak"],
                        "healthy_frames": healthy_frames,
                        "policy_name": f"train_step_{self.num_timesteps} ep{episode}",
                    },
                )
                frames.append(frame)

                if terminated or truncated:
                    episode += 1
                    obs, info = env.reset(seed=self.seed + self.num_timesteps + episode)
                    episode_return = 0.0
                    healthy_frames = 0
                    step = 0
                    continue

                step += 1
                if step >= self.max_episode_steps:
                    episode += 1
                    obs, info = env.reset(seed=self.seed + self.num_timesteps + episode)
                    episode_return = 0.0
                    healthy_frames = 0
                    step = 0
        finally:
            env.close()

        if frames:
            output = self.video_dir / f"step_{self.num_timesteps:09d}.mp4"
            iio.imwrite(output, frames, fps=self.fps)
            print(f"Wrote training video {output}")
            self._upload_to_github(output)

    def _upload_to_github(self, output: pathlib.Path) -> None:
        if not self.github_repo:
            return
        if not self.github_token:
            print("Skipping GitHub upload: GITHUB_TOKEN is not set.")
            return

        tag = self.github_release_tag or "ppo-7link-training-videos"
        release = self._get_or_create_release(tag)
        if release is None:
            return

        asset_name = output.name
        for asset in release.get("assets", []):
            if asset.get("name") == asset_name:
                print(f"Skipping GitHub upload: release asset already exists: {asset_name}")
                return

        upload_url = release["upload_url"].split("{", 1)[0]
        url = f"{upload_url}?{urllib.parse.urlencode({'name': asset_name})}"
        data = output.read_bytes()
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "video/mp4",
                "Content-Length": str(len(data)),
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status not in (200, 201):
                    print(f"GitHub upload returned HTTP {response.status} for {asset_name}")
                    return
            print(f"Uploaded training video to GitHub release {tag}: {asset_name}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"GitHub upload failed for {asset_name}: HTTP {exc.code} {body}")
        except OSError as exc:
            print(f"GitHub upload failed for {asset_name}: {exc}")

    def _get_or_create_release(self, tag: str) -> dict | None:
        existing = self._github_json(f"https://api.github.com/repos/{self.github_repo}/releases/tags/{tag}")
        if existing is not None:
            return existing

        body = {
            "tag_name": tag,
            "name": "PPO 7-link training videos",
            "body": "Automatically uploaded MuJoCo PPO training preview videos.",
            "draft": False,
            "prerelease": False,
        }
        return self._github_json(
            f"https://api.github.com/repos/{self.github_repo}/releases",
            method="POST",
            body=body,
        )

    def _github_json(self, url: str, method: str = "GET", body: dict | None = None) -> dict | None:
        import json

        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            body_text = exc.read().decode("utf-8", errors="replace")
            print(f"GitHub API request failed: HTTP {exc.code} {body_text}")
            return None
        except OSError as exc:
            print(f"GitHub API request failed: {exc}")
            return None
