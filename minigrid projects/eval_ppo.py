import argparse
import json
from collections import deque
import gymnasium as gym
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.simple_keydoor_env import register_env
from envs.wrappers import BeliefMapObsWrapper, ReducedActionSpace
from gymnasium.wrappers import FlattenObservation

from gymnasium.wrappers import RecordVideo


class MiniGridCNN(BaseFeaturesExtractor):
    """
    Feature extractor for MiniGrid 7x7 observations.
    Works with channel-first (C,H,W) after VecTransposeImage.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # C

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),               # 7x7 -> 4x4
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),               # 4x4 -> 2x2
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def bfs_distance(grid, start, goal, width, height, passable_types=None):
    """
    BFS shortest path distance on a MiniGrid grid.
    passable_types: set of cell.type strings treated as walkable (None = empty cell).
    Returns the step count, or None if unreachable.
    """
    if passable_types is None:
        passable_types = {"key", "goal", "door"}
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == goal:
            return dist
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            npos = (nx, ny)
            if npos in visited or not (0 <= nx < width and 0 <= ny < height):
                continue
            cell = grid.get(nx, ny)
            cell_type = None if cell is None else cell.type
            if cell_type is None or cell_type in passable_types:
                visited.add(npos)
                queue.append((npos, dist + 1))
    return None


def compute_kl_divergence(base_env) -> float:
    """
    Per-cell Bernoulli KL divergence: KL(true_wall || belief_wall).
    Measures how well the belief map matches the ground truth.
    Lower is better (0.0 = perfect belief).
    """
    eps = 1e-6
    true_wall = np.zeros((base_env.height, base_env.width), dtype=np.float32)
    for y in range(base_env.height):
        for x in range(base_env.width):
            cell = base_env.grid.get(x, y)
            if cell is not None and cell.type == "wall":
                true_wall[y, x] = 1.0

    p = np.clip(true_wall, eps, 1 - eps)
    q = np.clip(base_env.belief.wall, eps, 1 - eps)
    kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return float(kl.mean())


def compute_optimal_steps(base_env) -> int:
    """
    BFS lower bound for the episode.
    Curriculum mode (no key/door): just start -> goal.
    Full mode: start -> key -> door -> goal.
    """
    g = base_env.grid
    w, h = base_env.width, base_env.height
    start = tuple(base_env.agent_start_pos)
    goal  = tuple(base_env.goal_pos)

    # Check if key/door exist on the grid
    key_pos  = getattr(base_env, "key_pos",  None)
    door_pos = getattr(base_env, "door_pos", None)

    passable = {"key", "goal", "door"}

    key_obj  = g.get(*key_pos)  if key_pos  else None
    door_obj = g.get(*door_pos) if door_pos else None

    has_key  = key_obj  is not None and key_obj.type  == "key"
    has_door = door_obj is not None and door_obj.type == "door"

    if has_key and has_door:
        key  = tuple(key_pos)
        door = tuple(door_pos)
        d1 = bfs_distance(g, start, key,  w, h, passable)
        d2 = bfs_distance(g, key,   door, w, h, passable)
        d3 = bfs_distance(g, door,  goal, w, h, passable)
        if None in (d1, d2, d3):
            return None
        return d1 + d2 + d3
    else:
        # Curriculum: direct start -> goal
        d = bfs_distance(g, start, goal, w, h, passable)
        return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true",
                        help="Show live game + belief map window via matplotlib (safe, no pygame GIL crash)")

    parser.add_argument("--video", action="store_true", help="Save videos to disk")
    parser.add_argument("--video_dir", type=str, default="videos", help="Folder to save mp4 videos")
    parser.add_argument("--video_prefix", type=str, default="ppo_simplekeydoor", help="Video filename prefix")
    parser.add_argument("--view_size", type=int, default=None, help="Agent view size (overrides level config)")
    parser.add_argument("--grid_size", type=int, default=None, help="Grid size (overrides level config)")
    parser.add_argument("--num_walls", type=int, default=None, help="Extra walls (overrides level config)")
    parser.add_argument("--level", type=str, default=None,
                        help="Path to level JSON file. Auto-detected from run dir if not set.")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Seconds to pause between frames when rendering (default: 0.05)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic). "
                             "Useful to diagnose train/eval gap on old models.")
    parser.add_argument("--noise_prob", type=float, default=None,
                        help="Override noise level for eval (default: matches training config). "
                             "Use this to test model at different noise levels for graphs.")

    args = parser.parse_args()

    from pathlib import Path

    # Load level config: try --level first, then look inside the model's run folder
    level_cfg = {}
    level_path = args.level
    if level_path is None:
        candidate = Path(args.model).parent / "level.json"
        if candidate.exists():
            level_path = str(candidate)
    if level_path:
        with open(level_path, "r", encoding="utf-8") as f:
            level_cfg = json.load(f)
        print(f"Loaded level config: {level_path}")

    # Load training config from the run folder (auto-detects grid_size, view_size, etc.)
    train_cfg = {}
    train_cfg_path = Path(args.model).parent / "config.json"
    if train_cfg_path.exists():
        with open(train_cfg_path, "r", encoding="utf-8") as f:
            train_cfg = json.load(f)
        print(f"Loaded training config: {train_cfg_path}")

    # Priority: CLI args > level.json > config.json > hardcoded defaults
    view_size  = args.view_size  if args.view_size  is not None else level_cfg.get("view_size",  train_cfg.get("view_size",  5))
    grid_size  = args.grid_size  if args.grid_size  is not None else level_cfg.get("grid_size",  train_cfg.get("grid_size",  8))
    num_walls  = args.num_walls  if args.num_walls  is not None else level_cfg.get("num_walls",  train_cfg.get("num_walls",  0))
    # CLI --noise_prob overrides config — needed for graph generation across noise levels
    noise_prob = args.noise_prob if args.noise_prob is not None else train_cfg.get("noise_prob", 0.0)

    register_env()

    # Always use rgb_array so MiniGrid never touches pygame from inside DummyVecEnv.
    # We render manually via matplotlib in the main thread.
    render_mode = "rgb_array" if (args.render or args.video) else None

    def make_eval_env():
        from envs.simple_keydoor_env import SensorConfig
        sensor_cfg = SensorConfig(noise_prob=noise_prob)  # match training noise
        e = gym.make(
            "SimpleKeyDoor-v0",
            render_mode=render_mode,
            size=grid_size,
            num_extra_walls=num_walls,
            agent_view_size=view_size,
            sensor_cfg=sensor_cfg,
        )
        e = ReducedActionSpace(e)
        e = BeliefMapObsWrapper(e)
        e = FlattenObservation(e)

        if args.video:
            e = RecordVideo(
                e,
                video_folder=args.video_dir,
                episode_trigger=lambda ep_id: True,
                name_prefix=args.video_prefix,
                disable_logger=True,
            )
        return e

    env = DummyVecEnv([make_eval_env])

    model = PPO.load(args.model, env=env, custom_objects={})

    ep_rewards = []
    ep_steps = []
    ep_kl = []
    ep_optimality = []
    successes = 0

    # ------------------------------------------------------------------ #
    # Render window — one figure with game frame + 4 belief heatmaps
    # All matplotlib calls stay in the main thread; pygame is rgb_array only.
    # ------------------------------------------------------------------ #
    render_fig = None
    ax_game = None
    belief_ims = []
    belief_agent_dots = []

    if args.render and MATPLOTLIB_AVAILABLE:
        plt.ion()
        # Layout: top row = game frame (spans 2 cols) | right 2 cols = wall/key
        #         bottom row = door | goal | (empty) | (empty)
        render_fig = plt.figure(figsize=(14, 7))
        render_fig.suptitle("PPO Agent — Live View", fontsize=13, fontweight="bold")

        gs = render_fig.add_gridspec(2, 4, hspace=0.4, wspace=0.35)

        # Left panel: actual game frame (2 rows tall)
        ax_game = render_fig.add_subplot(gs[:, :2])
        ax_game.set_title("Game (rgb_array)", fontsize=10)
        ax_game.axis("off")
        game_im = ax_game.imshow(np.zeros((grid_size * 32, grid_size * 32, 3), dtype=np.uint8))

        channel_titles = ["Wall probability", "Key probability",
                          "Door probability", "Goal probability"]
        channel_cmaps  = ["Reds", "YlOrBr", "Blues", "Greens"]
        belief_axes_positions = [gs[0, 2], gs[0, 3], gs[1, 2], gs[1, 3]]

        for pos, title, cmap in zip(belief_axes_positions, channel_titles, channel_cmaps):
            ax = render_fig.add_subplot(pos)
            im = ax.imshow(
                np.full((grid_size, grid_size), 0.5),
                vmin=0.0, vmax=1.0,
                cmap=cmap, interpolation="nearest", origin="upper",
            )
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            render_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            dot_agent, = ax.plot([], [], "wx", markersize=9,  markeredgewidth=2, label="agent")
            dot_goal,  = ax.plot([], [], "y*", markersize=11, label="goal")
            dot_key,   = ax.plot([], [], "g^", markersize=8,  label="key")
            dot_door,  = ax.plot([], [], "bs", markersize=8,  label="door")

            belief_ims.append(im)
            belief_agent_dots.append((dot_agent, dot_goal, dot_key, dot_door))

        handles = [
            mpatches.Patch(color="none", label="white X = agent"),
            mpatches.Patch(color="none", label="yellow ★ = goal"),
            mpatches.Patch(color="none", label="green ▲ = key"),
            mpatches.Patch(color="none", label="blue ■ = door"),
        ]
        render_fig.legend(handles=handles, loc="lower center", ncol=4,
                          fontsize=8, framealpha=0.7)
        plt.show(block=False)
        plt.pause(0.01)

    elif args.render and not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not installed — render window disabled. "
              "Run: pip install matplotlib")

    def update_render_window(base_env, rgb_frame):
        """Refresh game frame + belief heatmaps. Must be called from main thread."""
        if render_fig is None:
            return

        # Update game frame
        if rgb_frame is not None and ax_game is not None:
            game_im.set_data(rgb_frame)
            game_im.set_extent([0, rgb_frame.shape[1], rgb_frame.shape[0], 0])
            ax_game.set_xlim(0, rgb_frame.shape[1])
            ax_game.set_ylim(rgb_frame.shape[0], 0)

        channels = [
            base_env.belief.wall,
            base_env.belief.key,
            base_env.belief.door,
            base_env.belief.goal,
        ]
        ax_pos   = int(base_env.agent_pos[0]), int(base_env.agent_pos[1])
        goal_pos = int(base_env.goal_pos[0]),  int(base_env.goal_pos[1])

        key_pos  = getattr(base_env, "key_pos",  None)
        door_pos = getattr(base_env, "door_pos", None)
        kx, ky = (int(key_pos[0]),  int(key_pos[1]))  if key_pos  else (None, None)
        dx, dy = (int(door_pos[0]), int(door_pos[1])) if door_pos else (None, None)

        for im, (d_agent, d_goal, d_key, d_door), data in zip(
                belief_ims, belief_agent_dots, channels):
            im.set_data(data)
            d_agent.set_data([ax_pos[0]],  [ax_pos[1]])
            d_goal.set_data( [goal_pos[0]], [goal_pos[1]])
            if kx is not None:
                d_key.set_data([kx], [ky])
            if dx is not None:
                d_door.set_data([dx], [dy])

        render_fig.canvas.draw_idle()
        plt.pause(args.delay)

    # ------------------------------------------------------------------ #

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total = 0.0
        step = 0

        base = env.envs[0].unwrapped
        optimal = compute_optimal_steps(base)

        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "toggle"}
            a = int(action) if np.isscalar(action) else int(action[0])

            base = env.envs[0].unwrapped
            agent_pos = tuple(base.agent_pos)
            goal_pos  = tuple(base.goal_pos)

            door_obj     = base.grid.get(*base.door_pos) if hasattr(base, "door_pos") else None
            door_is_open = (door_obj is not None and door_obj.type == "door" and door_obj.is_open)
            door_is_locked = (door_obj is not None and door_obj.type == "door" and door_obj.is_locked)
            carrying     = None if base.carrying is None else base.carrying.type

            print(
                f"pos={agent_pos} | goal={goal_pos} | "
                f"door_open={door_is_open} | door_locked={door_is_locked} | "
                f"carrying={carrying} | action={a}:{action_names.get(a)}"
            )

            obs, reward, dones, infos = env.step(action)
            total += float(reward[0])
            done = bool(dones[0])
            step += 1

            # Grab rgb frame AFTER step (main thread, safe)
            if args.render and render_fig is not None:
                try:
                    rgb_frame = env.envs[0].unwrapped.render()
                except Exception:
                    rgb_frame = None
                update_render_window(env.envs[0].unwrapped, rgb_frame)

        # --- Per-episode metrics ---
        base = env.envs[0].unwrapped
        kl = compute_kl_divergence(base)
        ep_kl.append(kl)

        if optimal is not None and step > 0:
            optimality = optimal / step
            ep_optimality.append(optimality)
        else:
            optimality = None

        success = infos[0].get("is_success", False)
        ep_rewards.append(total)
        ep_steps.append(step)
        if success:
            successes += 1

        print(f"Episode {ep + 1} | steps={step} | reward={total:.3f} | "
              f"success={success} | KL={kl:.4f} | "
              f"optimality={f'{optimality:.3f}' if optimality is not None else 'N/A'}")

        if done:
            final_pos = tuple(infos[0].get("agent_pos", (0, 0)))
            goal_pos_info = tuple(infos[0].get("goal_pos", (0, 0)))
            print(f"  FINAL POS: {final_pos} | GOAL POS: {goal_pos_info}")

    env.close()
    if render_fig is not None:
        plt.ioff()
        plt.close(render_fig)

    if args.video:
        print(f"Videos saved to: {args.video_dir}")

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Episodes        : {args.episodes}")
    print(f"Success rate    : {successes / args.episodes:.2%}")
    print(f"Mean reward     : {np.mean(ep_rewards):.3f}  (std {np.std(ep_rewards):.3f})")
    print(f"Mean steps      : {np.mean(ep_steps):.1f}  (std {np.std(ep_steps):.1f})")
    print(f"Mean KL div     : {np.mean(ep_kl):.4f}  (belief vs true wall map; lower = better)")
    if ep_optimality:
        print(f"Path optimality : {np.mean(ep_optimality):.3f}  (optimal/actual steps; 1.0 = perfect)")
    print("=========================================")


if __name__ == "__main__":
    main()
