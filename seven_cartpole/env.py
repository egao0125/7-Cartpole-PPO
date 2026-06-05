from __future__ import annotations

import pathlib
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from seven_cartpole.rewards import RewardWeights, shaped_reward


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = ROOT / "seven_pendulum_cartpole.xml"


@dataclass(frozen=True)
class EnvConfig:
    xml_path: pathlib.Path = DEFAULT_XML_PATH
    frame_skip: int = 10
    max_episode_steps: int = 2000
    init_angle_noise: float = 0.025
    init_velocity_noise: float = 0.005
    cart_limit: float = 2.15
    terminate_on_low_tip_after_steps: int = 150
    min_tip_height: float = 0.45
    reward_weights: RewardWeights = RewardWeights()


class SevenPendulumCartpoleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        width: int = 1280,
        height: int = 720,
        camera: str = "side",
    ):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.camera = camera

        self.model = mujoco.MjModel.from_xml_path(str(self.config.xml_path))
        self.data = mujoco.MjData(self.model)
        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")

        self.n_links = self.model.nq - 1
        if self.n_links != 7:
            raise ValueError(f"Expected 7 hinge links, found {self.n_links}")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

        self._renderer = None
        self._step_count = 0
        self._last_tip_xz = np.zeros(2, dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        angle_noise = float(options.get("init_angle_noise", self.config.init_angle_noise))
        velocity_noise = float(options.get("init_velocity_noise", self.config.init_velocity_noise))

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[1:] = self.np_random.normal(0.0, angle_noise, size=self.n_links)
        self.data.qvel[:] = self.np_random.normal(0.0, velocity_noise, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._last_tip_xz = self._tip_xz()
        return self._get_obs(), self._info(action=0.0, reward_terms={})

    def step(self, action):
        action_scalar = float(np.clip(np.asarray(action, dtype=np.float32)[0], -1.0, 1.0))
        self.data.ctrl[0] = action_scalar

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, reward_terms = self._reward(action_scalar)
        terminated = self._terminated()
        truncated = self._step_count >= self.config.max_episode_steps
        info = self._info(action=action_scalar, reward_terms=reward_terms)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self):
        cart_x = float(self.data.qpos[0])
        cart_v = float(self.data.qvel[0])
        angles = np.asarray(self.data.qpos[1:], dtype=np.float64)
        angular_velocities = np.asarray(self.data.qvel[1:], dtype=np.float64)
        tip_xz = self._tip_xz()
        tip_velocity_xz = (tip_xz - self._last_tip_xz) / (self.model.opt.timestep * self.config.frame_skip)
        self._last_tip_xz = tip_xz.copy()

        obs = np.concatenate(
            [
                np.array([cart_x, cart_v], dtype=np.float64),
                np.sin(angles),
                np.cos(angles),
                angular_velocities,
                tip_xz,
                tip_velocity_xz,
            ]
        )
        return obs.astype(np.float32)

    def _reward(self, action: float):
        return shaped_reward(
            tip_height=self.tip_height,
            cart_x=float(self.data.qpos[0]),
            cart_v=float(self.data.qvel[0]),
            angles=np.asarray(self.data.qpos[1:], dtype=np.float64),
            angular_velocities=np.asarray(self.data.qvel[1:], dtype=np.float64),
            action=action,
            weights=self.config.reward_weights,
        )

    def _terminated(self):
        if abs(float(self.data.qpos[0])) > self.config.cart_limit:
            return True
        if self._step_count > self.config.terminate_on_low_tip_after_steps and self.tip_height < self.config.min_tip_height:
            return True
        return False

    def _info(self, *, action: float, reward_terms: dict[str, float]):
        return {
            "step": self._step_count,
            "sim_time": float(self.data.time),
            "action": float(action),
            "cart_x": float(self.data.qpos[0]),
            "cart_v": float(self.data.qvel[0]),
            "tip_height": self.tip_height,
            "reward_terms": reward_terms,
        }

    def _tip_xz(self):
        tip = self.data.site_xpos[self.tip_site_id]
        return np.array([tip[0], tip[2]], dtype=np.float64)

    @property
    def tip_height(self):
        return float(self.data.site_xpos[self.tip_site_id, 2])
