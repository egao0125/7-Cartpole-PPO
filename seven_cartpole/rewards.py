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
