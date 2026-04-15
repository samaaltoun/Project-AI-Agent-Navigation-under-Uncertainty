"""
eval_belief_viz.py — Real-time visualization of the agent's belief map.

Displays two side-by-side panels:
  LEFT:  The MiniGrid environment (what's actually happening)
  RIGHT: The belief map entropy heatmap (what the agent "thinks")

Usage
-----
  python eval_belief_viz.py --model runs/<folder>/model.zip
  python eval_belief_viz.py --model runs/<folder>/model.zip --noise 0.3 --episodes 5
  python eval_belief_viz.py --model runs/<folder>/model.zip --speed 0.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import ImgObsWrapper

from envs.simple_keydoor_env import register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper, BeliefObsWrapper
from envs.extractors import MiniGridCNN
from envs.belief_map import BeliefMap


ACTION_NAMES = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "toggle"}


def _render_grid_text(base) -> np.ndarray:
    grid = base.grid
    w, h = grid.width, grid.height
    cell_size = 20
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    colors = {
        "wall":  (100, 100, 100),
        "empty": (0, 0, 0),
        "floor": (20, 20, 20),
        "key":   (255, 255, 0),
        "door":  (0, 100, 255),
        "goal":  (0, 255, 0),
        "lava":  (255, 50, 0),
    }

    for y in range(h):
        for x in range(w):
            cell = grid.get(x, y)
            if cell is None:
                color = colors["empty"]
            else:
                color = colors.get(cell.type, (128, 128, 128))
                if cell.type == "door":
                    if cell.is_open:
                        color = (100, 200, 255)
                    elif cell.is_locked:
                        color = (0, 50, 150)

            y0, y1 = y * cell_size, (y + 1) * cell_size
            x0, x1 = x * cell_size, (x + 1) * cell_size
            img[y0:y1, x0:x1] = color

    # Draw agent
    ax, ay = int(base.agent_pos[0]), int(base.agent_pos[1])
    cy, cx = ay * cell_size + cell_size // 2, ax * cell_size + cell_size // 2
    r = cell_size // 3
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                py, px = cy + dy, cx + dx
                if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                    img[py, px] = (255, 0, 0)

    # Draw direction indicator
    dir_vec = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    ddx, ddy = dir_vec[base.agent_dir]
    for i in range(1, cell_size // 2):
        px = cx + ddx * i
        py = cy + ddy * i
        if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
            img[py, px] = (255, 255, 255)

    return img


def main():
    parser = argparse.ArgumentParser(description="Real-time belief map visualization")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--n_lava", type=int, default=None)
    parser.add_argument("--speed", type=float, default=0.1)
    args = parser.parse_args()

    # Load config
    config_path = Path(args.model).parent / "config.json"
    saved = {}
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))

    if args.grid_size is None:
        args.grid_size = saved.get("grid_size", 15)
    if args.n_lava is None:
        args.n_lava = saved.get("n_lava", 0)
    if args.noise is None:
        args.noise = saved.get("noise_prob", 0.0)

    use_belief_obs = "decay" in saved
    decay = saved.get("decay", 0.0)
    tpr = saved.get("tpr", None)

    register_env()

    # We need a reference to the belief wrapper for visualization
    belief_wrapper_ref = [None]

    def make_env():
        e = gym.make("SimpleKeyDoor-v0", render_mode=None,
                     size=args.grid_size, n_lava=args.n_lava)
        e = ReducedActionSpace(e)
        if args.noise > 0.0:
            e = SensorNoiseWrapper(e, noise_prob=args.noise)
        if use_belief_obs:
            e = BeliefObsWrapper(e, grid_size=args.grid_size,
                                 noise_prob=args.noise, tpr=tpr, decay=decay)
            belief_wrapper_ref[0] = e
        else:
            e = ImgObsWrapper(e)
        return e

    env = DummyVecEnv([make_env])
    if not use_belief_obs:
        env = VecTransposeImage(env)  # BeliefObsWrapper already outputs (C,H,W)

    model = PPO.load(args.model, env=env,
                     custom_objects={"features_extractor_class": MiniGridCNN})

    # DummyVecEnv (no VecTransposeImage) → env.envs[0]
    # VecTransposeImage wraps DummyVecEnv  → env.venv.envs[0]
    if hasattr(env, "venv"):
        base = env.venv.envs[0].unwrapped
    else:
        base = env.envs[0].unwrapped

    # Get belief map — from wrapper or standalone
    if belief_wrapper_ref[0] is not None:
        bmap = belief_wrapper_ref[0].bmap
    else:
        bmap = BeliefMap(grid_size=args.grid_size, noise_prob=args.noise)

    # Set up the figure
    plt.ion()
    fig, (ax_grid, ax_belief) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Agent Navigation under Uncertainty", fontsize=14)
    max_entropy = np.log2(10)

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        step = 0

        # Reset standalone belief map (wrapper handles its own reset)
        if belief_wrapper_ref[0] is None:
            bmap.reset()
            bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            a = int(action) if np.isscalar(action) else int(action[0])

            has_key = base.carrying is not None and base.carrying.type == "key"
            door = base.grid.get(*base.door_pos)
            door_open = door is not None and door.type == "door" and door.is_open

            # ── LEFT PANEL: Grid ──
            ax_grid.clear()
            grid_img = _render_grid_text(base)
            ax_grid.imshow(grid_img)
            ax_grid.set_title(
                f"Episode {ep+1} | Step {step} | Action: {ACTION_NAMES.get(a, '?')}\n"
                f"Carrying: {'key' if has_key else 'none'} | Door: {'open' if door_open else 'locked'}",
                fontsize=10
            )
            ax_grid.axis("off")

            # ── RIGHT PANEL: Belief entropy heatmap ──
            ax_belief.clear()
            entropy = bmap.entropy_map()
            im = ax_belief.imshow(entropy.T, cmap="magma_r", vmin=0,
                                  vmax=max_entropy, interpolation="nearest")
            agx, agy = int(base.agent_pos[0]), int(base.agent_pos[1])
            ax_belief.plot(agx, agy, "wo", markersize=8,
                          markeredgecolor="black", markeredgewidth=1.5)
            kx, ky = int(base.key_pos[0]), int(base.key_pos[1])
            dx, dy = int(base.door_pos[0]), int(base.door_pos[1])
            gx, gy = int(base.goal_pos[0]), int(base.goal_pos[1])
            if not has_key:
                ax_belief.plot(kx, ky, "y*", markersize=12,
                              markeredgecolor="black", markeredgewidth=0.5)
            ax_belief.plot(dx, dy, "bs", markersize=8,
                          markeredgecolor="black", markeredgewidth=0.5)
            ax_belief.plot(gx, gy, "g^", markersize=10,
                          markeredgecolor="black", markeredgewidth=0.5)

            mean_h = bmap.mean_entropy()
            kl = bmap.kl_from_truth(base.grid)
            ax_belief.set_title(
                f"Belief Map Entropy (noise={args.noise})\n"
                f"Mean entropy: {mean_h:.2f} bits | KL from truth: {kl:.2f} bits",
                fontsize=10
            )
            ax_belief.set_xlabel("x")
            ax_belief.set_ylabel("y")

            plt.tight_layout()
            plt.pause(args.speed)

            # Step the environment
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            step += 1

            # Update standalone belief map (wrapper handles its own update)
            if belief_wrapper_ref[0] is None:
                bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)

        is_success = infos[0].get("is_success", False)
        print(f"Episode {ep+1}: {'SUCCESS' if is_success else 'FAIL'} in {step} steps")

        ax_grid.set_title(
            f"Episode {ep+1} FINISHED — {'SUCCESS' if is_success else 'FAIL'} in {step} steps",
            fontsize=11, color="green" if is_success else "red"
        )
        plt.pause(2.0)

    env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
