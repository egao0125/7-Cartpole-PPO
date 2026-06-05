from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def draw_hud(frame: np.ndarray, stats: dict) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(img)
    lines = [
        "MuJoCo 7-link PPO cartpole",
        f"step: {stats['step']}   sim: {stats['sim_time']:.2f}s   action: {stats['action']:+.2f}",
        f"return: {stats['episode_return']:.3f}   reward: {stats['reward']:.3f}",
        f"tip height: {stats['tip_height']:.3f}   cart x: {stats['cart_x']:+.3f}",
        f"uprightness: {stats['uprightness']:.3f}   healthy: {stats['is_healthy']}",
        f"healthy frames: {stats['healthy_frames']}   policy: {stats['policy_name']}",
    ]
    pad = 14
    line_h = 19
    width = max(draw.textlength(line) for line in lines) + pad * 2
    height = line_h * len(lines) + pad
    draw.rounded_rectangle((12, 12, 12 + width, 12 + height), radius=6, fill=(0, 0, 0, 170))
    for i, line in enumerate(lines):
        draw.text((12 + pad, 19 + i * line_h), line, fill=(245, 245, 245, 255))
    return np.asarray(img.convert("RGB"))
