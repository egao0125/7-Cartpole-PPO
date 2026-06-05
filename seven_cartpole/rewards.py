from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardWeights:
    tip_height: float = 2.0
    upright: float = 0.5
    cart_position: float = 0.3
    cart_velocity: float = 0.02
    action: float = 0.01
    angular_velocity: float = 0.002
    survival: float = 0.02


@dataclass(frozen=True)
class SwingUpRewardWeights:
    height: float = 4.0
    upright: float = 1.0
    height_progress: float = 12.0
    upward_tip_velocity: float = 0.4
    cart_motion_when_low: float = 0.05
    action_pump_when_low: float = 0.01
    healthy_hold: float = 2.0
    cart_position: float = 0.15
    action_when_low: float = 0.0
    action_when_high: float = 0.04
    cart_velocity_when_high: float = 0.06
    tip_speed_when_high: float = 0.18
    angular_velocity_when_low: float = 0.001
    angular_velocity_when_high: float = 0.04


def shaped_reward(
    *,
    tip_height: float,
    cart_x: float,
    cart_v: float,
    angles: np.ndarray,
    angular_velocities: np.ndarray,
    action: float,
    healthy: bool,
    weights: RewardWeights = RewardWeights(),
) -> tuple[float, dict[str, float]]:
    global_angles = np.cumsum(angles)
    upright = float(np.mean(np.cos(global_angles)))
    angular_velocity_cost = float(np.sum(np.square(angular_velocities)))
    action_cost = float(action * action)
    cart_velocity_cost = float(cart_v * cart_v)

    terms = {
        "tip_height": weights.tip_height * float(tip_height),
        "upright": weights.upright * upright,
        "cart_position": -weights.cart_position * abs(float(cart_x)),
        "cart_velocity": -weights.cart_velocity * cart_velocity_cost,
        "action": -weights.action * action_cost,
        "angular_velocity": -weights.angular_velocity * angular_velocity_cost,
        "alive": weights.survival if healthy else 0.0,
    }
    return float(sum(terms.values())), terms


def swingup_reward(
    *,
    tip_height: float,
    previous_tip_height: float,
    min_tip_height: float,
    max_tip_height: float,
    cart_x: float,
    cart_v: float,
    tip_speed: float,
    angles: np.ndarray,
    angular_velocities: np.ndarray,
    action: float,
    healthy: bool,
    weights: SwingUpRewardWeights = SwingUpRewardWeights(),
) -> tuple[float, dict[str, float]]:
    global_angles = np.cumsum(angles)
    upright = float(np.mean(np.cos(global_angles)))
    height_range = max(max_tip_height - min_tip_height, 1e-6)
    normalized_height = float(np.clip((tip_height - min_tip_height) / height_range, 0.0, 1.0))
    height_delta = float(tip_height - previous_tip_height)
    upward_tip_velocity = max(height_delta, 0.0)
    low_height = 1.0 - normalized_height

    near_upright = normalized_height > 0.75
    action_cost = float(action * action)
    cart_velocity_cost = float(cart_v * cart_v)
    tip_speed_cost = float(tip_speed * tip_speed)
    angular_velocity_cost = float(np.sum(np.square(angular_velocities)))

    cart_position_weight = weights.cart_position if near_upright else 0.0
    action_weight = weights.action_when_high if near_upright else weights.action_when_low
    cart_velocity_weight = weights.cart_velocity_when_high if near_upright else 0.0
    tip_speed_weight = weights.tip_speed_when_high if near_upright else 0.0
    angular_velocity_weight = (
        weights.angular_velocity_when_high if near_upright else weights.angular_velocity_when_low
    )

    terms = {
        "height": weights.height * normalized_height,
        "upright": weights.upright * ((upright + 1.0) * 0.5),
        "height_progress": weights.height_progress * upward_tip_velocity * low_height,
        "upward_tip_velocity": weights.upward_tip_velocity * upward_tip_velocity * low_height,
        "cart_motion_when_low": weights.cart_motion_when_low * abs(float(cart_v)) * low_height,
        "action_pump_when_low": weights.action_pump_when_low * abs(float(action)) * low_height,
        "healthy_hold": weights.healthy_hold if healthy else 0.0,
        "cart_position": -cart_position_weight * abs(float(cart_x)),
        "action": -action_weight * action_cost,
        "cart_velocity": -cart_velocity_weight * cart_velocity_cost,
        "tip_speed": -tip_speed_weight * tip_speed_cost,
        "angular_velocity": -angular_velocity_weight * angular_velocity_cost,
    }
    return float(sum(terms.values())), terms
