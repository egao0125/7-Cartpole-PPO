from __future__ import annotations

import numpy as np


def heuristic_action_from_obs(obs: np.ndarray) -> np.ndarray:
    cart_x = float(obs[0])
    cart_v = float(obs[1])
    sin_angles = obs[2:9]
    angular_velocities = obs[16:23]

    weights = np.linspace(1.0, 0.25, len(sin_angles))
    lean = float(np.dot(weights, sin_angles))
    angular_rate = float(np.dot(weights, angular_velocities))
    force = 2.6 * lean + 0.35 * angular_rate - 0.9 * cart_x - 0.18 * cart_v
    return np.array([np.clip(force, -1.0, 1.0)], dtype=np.float32)
