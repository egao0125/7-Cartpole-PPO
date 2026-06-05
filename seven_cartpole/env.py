from __future__ import annotations

import pathlib
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from seven_cartpole.rewards import RewardWeights, SwingUpRewardWeights, shaped_reward, swingup_reward


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = ROOT / "seven_pendulum_cartpole.xml"


@dataclass(frozen=True)
class EnvConfig:
    xml_path: pathlib.Path = DEFAULT_XML_PATH
    task: str = "swingup"
    frame_skip: int = 10
    max_episode_steps: int = 2000
    init_angle_noise: float = 0.025
    init_velocity_noise: float = 0.005
    cart_limit: float = 2.15
    health_warmup_steps: int = 50
    min_tip_height: float = 3.0
    min_uprightness: float = 0.65
    max_healthy_tip_speed: float = 0.8
    max_healthy_cart_speed: float = 0.7
    max_healthy_angular_velocity: float = 3.0
    max_safe_tip_speed: float = 8.0
    max_safe_angular_velocity: float = 20.0
    success_hold_steps: int = 100
    swingup_stuck_warmup_steps: int = 150
    swingup_stuck_reset_steps: int = 125
    swingup_stuck_height_fraction: float = 0.25
    swingup_stuck_tip_speed: float = 0.08
    swingup_stuck_cart_speed: float = 0.03
    reward_weights: RewardWeights = RewardWeights()
    swingup_reward_weights: SwingUpRewardWeights = SwingUpRewardWeights()


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
        self._healthy_streak = 0
        self._previous_tip_height = 0.0
        self._reward_min_tip_height, self._reward_max_tip_height = self._tip_height_bounds()
        self._last_tip_xz = np.zeros(2, dtype=np.float64)
        self._last_tip_speed = 0.0
        self._swingup_stuck_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        angle_noise = float(options.get("init_angle_noise", self.config.init_angle_noise))
        velocity_noise = float(options.get("init_velocity_noise", self.config.init_velocity_noise))

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if self.config.task == "swingup":
            self.data.qpos[1] = np.pi
            self.data.qpos[2:] = 0.0
        elif self.config.task == "balance":
            self.data.qpos[1:] = 0.0
        else:
            raise ValueError(f"Unknown task: {self.config.task}")

        self.data.qpos[1:] += self.np_random.normal(0.0, angle_noise, size=self.n_links)
        self.data.qvel[:] = self.np_random.normal(0.0, velocity_noise, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._healthy_streak = 0
        self._previous_tip_height = self.tip_height
        self._last_tip_xz = self._tip_xz()
        self._last_tip_speed = 0.0
        self._swingup_stuck_steps = 0
        return self._get_obs(), self._info(action=0.0, reward_terms={})

    def step(self, action):
        action_scalar = float(np.clip(np.asarray(action, dtype=np.float32)[0], -1.0, 1.0))
        self.data.ctrl[0] = action_scalar
        self._previous_tip_height = self.tip_height

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, reward_terms = self._reward(action_scalar)
        health = self._health()
        self._healthy_streak = self._healthy_streak + 1 if health["is_healthy"] else 0
        self._update_swingup_stuck_state()
        terminated = self._terminated()
        truncated = self._step_count >= self.config.max_episode_steps
        info = self._info(action=action_scalar, reward_terms=reward_terms, truncated=truncated)
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
        self._last_tip_speed = float(np.linalg.norm(tip_velocity_xz))

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
        health = self._health()
        if self.config.task == "swingup":
            return swingup_reward(
                tip_height=self.tip_height,
                previous_tip_height=self._previous_tip_height,
                min_tip_height=self._reward_min_tip_height,
                max_tip_height=self._reward_max_tip_height,
                cart_x=float(self.data.qpos[0]),
                cart_v=float(self.data.qvel[0]),
                tip_speed=self._last_tip_speed,
                angles=np.asarray(self.data.qpos[1:], dtype=np.float64),
                angular_velocities=np.asarray(self.data.qvel[1:], dtype=np.float64),
                action=action,
                healthy=health["is_healthy"],
                weights=self.config.swingup_reward_weights,
            )

        return shaped_reward(
            tip_height=self.tip_height,
            cart_x=float(self.data.qpos[0]),
            cart_v=float(self.data.qvel[0]),
            angles=np.asarray(self.data.qpos[1:], dtype=np.float64),
            angular_velocities=np.asarray(self.data.qvel[1:], dtype=np.float64),
            action=action,
            healthy=health["is_healthy"],
            weights=self.config.reward_weights,
        )

    def _terminated(self):
        health = self._health()
        if not health["finite"] or not health["cart_ok"]:
            return True
        if self.config.task == "swingup":
            return not health["safe_velocity"] or self._swingup_is_stuck()
        if self._step_count > self.config.health_warmup_steps and not health["is_healthy"]:
            return True
        return False

    def _info(self, *, action: float, reward_terms: dict[str, float], truncated: bool = False):
        health = self._health()
        return {
            "step": self._step_count,
            "sim_time": float(self.data.time),
            "action": float(action),
            "cart_x": float(self.data.qpos[0]),
            "cart_v": float(self.data.qvel[0]),
            "tip_height": self.tip_height,
            "tip_speed": self._last_tip_speed,
            "max_angular_velocity": health["max_angular_velocity"],
            "uprightness": health["uprightness"],
            "is_healthy": health["is_healthy"],
            "healthy_streak": self._healthy_streak,
            "health": health,
            "swingup_stuck_steps": self._swingup_stuck_steps,
            "success": self._success(truncated=truncated, healthy=health["is_healthy"]),
            "reward_terms": reward_terms,
        }

    def _health(self):
        qpos = np.asarray(self.data.qpos, dtype=np.float64)
        qvel = np.asarray(self.data.qvel, dtype=np.float64)
        angles = qpos[1:]
        global_angles = np.cumsum(angles)
        uprightness = float(np.mean(np.cos(global_angles)))
        finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
        cart_ok = abs(float(qpos[0])) <= self.config.cart_limit
        max_angular_velocity = float(np.max(np.abs(qvel[1:]))) if qvel[1:].size else 0.0
        tip_ok = self.tip_height >= self.config.min_tip_height
        posture_ok = uprightness >= self.config.min_uprightness
        velocity_ok = (
            self._last_tip_speed <= self.config.max_healthy_tip_speed
            and abs(float(qvel[0])) <= self.config.max_healthy_cart_speed
            and max_angular_velocity <= self.config.max_healthy_angular_velocity
        )
        safe_velocity = (
            self._last_tip_speed <= self.config.max_safe_tip_speed
            and max_angular_velocity <= self.config.max_safe_angular_velocity
        )
        is_healthy = finite and cart_ok and tip_ok and posture_ok and velocity_ok
        return {
            "finite": finite,
            "cart_ok": cart_ok,
            "tip_ok": tip_ok,
            "posture_ok": posture_ok,
            "velocity_ok": velocity_ok,
            "safe_velocity": safe_velocity,
            "is_healthy": is_healthy,
            "uprightness": uprightness,
            "tip_height": self.tip_height,
            "tip_speed": self._last_tip_speed,
            "max_angular_velocity": max_angular_velocity,
            "cart_limit": self.config.cart_limit,
            "min_tip_height": self.config.min_tip_height,
            "min_uprightness": self.config.min_uprightness,
            "max_healthy_tip_speed": self.config.max_healthy_tip_speed,
            "max_healthy_cart_speed": self.config.max_healthy_cart_speed,
            "max_healthy_angular_velocity": self.config.max_healthy_angular_velocity,
            "max_safe_tip_speed": self.config.max_safe_tip_speed,
            "max_safe_angular_velocity": self.config.max_safe_angular_velocity,
        }

    def _success(self, *, truncated: bool, healthy: bool):
        if self.config.task == "swingup":
            return bool(self._healthy_streak >= self.config.success_hold_steps)
        return bool(truncated and healthy)

    def _update_swingup_stuck_state(self):
        if self.config.task != "swingup" or self._step_count < self.config.swingup_stuck_warmup_steps:
            self._swingup_stuck_steps = 0
            return

        if self._normalized_tip_height() > self.config.swingup_stuck_height_fraction:
            self._swingup_stuck_steps = 0
            return

        cart_speed = abs(float(self.data.qvel[0]))
        stuck = (
            self._last_tip_speed <= self.config.swingup_stuck_tip_speed
            and cart_speed <= self.config.swingup_stuck_cart_speed
        )
        self._swingup_stuck_steps = self._swingup_stuck_steps + 1 if stuck else 0

    def _swingup_is_stuck(self):
        return self._swingup_stuck_steps >= self.config.swingup_stuck_reset_steps

    def _normalized_tip_height(self):
        height_range = max(self._reward_max_tip_height - self._reward_min_tip_height, 1e-6)
        return float(np.clip((self.tip_height - self._reward_min_tip_height) / height_range, 0.0, 1.0))

    def _tip_xz(self):
        tip = self.data.site_xpos[self.tip_site_id]
        return np.array([tip[0], tip[2]], dtype=np.float64)

    def _tip_height_bounds(self):
        tmp = mujoco.MjData(self.model)
        tmp.qpos[:] = 0.0
        tmp.qvel[:] = 0.0
        mujoco.mj_forward(self.model, tmp)
        upright_height = float(tmp.site_xpos[self.tip_site_id, 2])

        tmp.qpos[:] = 0.0
        tmp.qvel[:] = 0.0
        tmp.qpos[1] = np.pi
        mujoco.mj_forward(self.model, tmp)
        hanging_height = float(tmp.site_xpos[self.tip_site_id, 2])
        return hanging_height, upright_height

    @property
    def tip_height(self):
        return float(self.data.site_xpos[self.tip_site_id, 2])
