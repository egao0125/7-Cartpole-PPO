#!/usr/bin/env python3
import argparse
import math
import pathlib
import time

import numpy as np
from PIL import Image, ImageDraw

try:
    import imageio.v3 as iio
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  python3 -m pip install -r requirements.txt\n"
        "On macOS, live viewer mode should be run with mjpython after install:\n"
        "  mjpython run_seven_cartpole.py --mode viewer"
    ) from exc


ROOT = pathlib.Path(__file__).resolve().parent
XML_PATH = ROOT / "seven_pendulum_cartpole.xml"


def reset_near_upright(model, data, rng):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[1:] = rng.normal(0.0, 0.025, size=7)
    mujoco.mj_forward(model, data)


def tip_height(model, data):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    return float(data.site_xpos[site_id, 2])


def heuristic_policy(model, data):
    """A deliberately simple baseline. Replace this with your trained policy."""
    cart_x = data.qpos[0]
    cart_v = data.qvel[0]
    angles = data.qpos[1:]
    vels = data.qvel[1:]

    # Push in the direction that counters the weighted lean of the chain.
    weights = np.linspace(1.0, 0.25, len(angles))
    lean = float(np.dot(weights, np.sin(angles)))
    angular_rate = float(np.dot(weights, vels))
    force = 2.6 * lean + 0.35 * angular_rate - 0.9 * cart_x - 0.18 * cart_v
    return float(np.clip(force, -1.0, 1.0))


def score(model, data):
    height = tip_height(model, data)
    cart_penalty = abs(float(data.qpos[0]))
    angle_penalty = float(np.mean(np.abs(np.sin(data.qpos[1:]))))
    return height - 0.25 * cart_penalty - 0.35 * angle_penalty


def draw_hud(frame, stats):
    img = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(img)
    lines = [
        "MuJoCo 7-link pendulum cartpole",
        f"step: {stats['step']}   sim: {stats['time']:.2f}s   ctrl: {stats['ctrl']:+.2f}",
        f"reward: {stats['reward']:.3f}   tip height: {stats['height']:.3f}",
        f"cart x: {stats['cart_x']:+.3f}   stable frames: {stats['stable']}",
    ]
    pad = 14
    line_h = 19
    width = max(draw.textlength(line) for line in lines) + pad * 2
    height = line_h * len(lines) + pad
    draw.rounded_rectangle((12, 12, 12 + width, 12 + height), radius=6, fill=(0, 0, 0, 160))
    for i, line in enumerate(lines):
        draw.text((12 + pad, 19 + i * line_h), line, fill=(245, 245, 245, 255))
    return np.asarray(img.convert("RGB"))


def render_video(args):
    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    reset_near_upright(model, data, rng)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames = []
    stable = 0
    steps_per_frame = max(1, round(1.0 / args.fps / model.opt.timestep))
    total_frames = round(args.duration * args.fps)

    for frame_idx in range(total_frames):
        ctrl = 0.0
        for _ in range(steps_per_frame):
            ctrl = heuristic_policy(model, data)
            data.ctrl[0] = ctrl
            mujoco.mj_step(model, data)

        height = tip_height(model, data)
        stable = stable + 1 if height > 1.25 and abs(data.qpos[0]) < 1.8 else 0

        renderer.update_scene(data, camera="side")
        frame = renderer.render()
        frame = draw_hud(
            frame,
            {
                "step": frame_idx * steps_per_frame,
                "time": data.time,
                "ctrl": ctrl,
                "reward": score(model, data),
                "height": height,
                "cart_x": float(data.qpos[0]),
                "stable": stable,
            },
        )
        frames.append(frame)

    output = ROOT / args.output
    iio.imwrite(output, frames, fps=args.fps)
    print(f"Wrote {output}")


def run_viewer(args):
    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    reset_near_upright(model, data, rng)
    manual = {"force": 0.0, "enabled": False}

    def key_callback(keycode):
        ch = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if ch == "r":
            reset_near_upright(model, data, rng)
        elif ch == "m":
            manual["enabled"] = not manual["enabled"]
        elif ch == "a":
            manual["force"] = -1.0
        elif ch == "d":
            manual["force"] = 1.0
        elif ch == "s":
            manual["force"] = 0.0

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

        last_print = 0.0
        while viewer.is_running():
            step_start = time.time()
            ctrl = manual["force"] if manual["enabled"] else heuristic_policy(model, data)
            data.ctrl[0] = ctrl
            mujoco.mj_step(model, data)

            if data.time - last_print > 0.25:
                last_print = data.time
                mode = "manual" if manual["enabled"] else "heuristic"
                print(
                    f"\rmode={mode} ctrl={ctrl:+.2f} reward={score(model, data):+.3f} "
                    f"height={tip_height(model, data):.3f} cart={data.qpos[0]:+.3f}",
                    end="",
                    flush=True,
                )

            viewer.sync()
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["video", "viewer"], default="video")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="seven_pendulum_cartpole.mp4")
    args = parser.parse_args()

    if args.mode == "video":
        render_video(args)
    else:
        run_viewer(args)


if __name__ == "__main__":
    main()
