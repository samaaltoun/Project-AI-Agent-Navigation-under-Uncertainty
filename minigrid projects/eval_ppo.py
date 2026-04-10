import argparse
import json
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import ImgObsWrapper

from envs.simple_keydoor_env import register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper
from envs.extractors import MiniGridCNN
from envs.astar import optimal_path_length

from gymnasium.wrappers import RecordVideo

from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # file-only backend; safe on headless machines
import matplotlib.pyplot as plt
from envs.belief_map import BeliefMap


def _plot_belief_metrics(
    entropy_histories: list,
    kl_histories: list,
    noise_prob: float,
    model_path: str,
) -> None:
    """
    Save a two-panel PNG alongside the model:
      left  — mean entropy of the belief map over episode steps
      right — mean KL divergence (belief vs ground truth) over episode steps

    The shaded band shows ± 1 standard deviation across episodes.
    """
    max_len = max(len(h) for h in entropy_histories)

    def _pad(hists):
        return np.array([h + [h[-1]] * (max_len - len(h)) for h in hists], dtype=float)

    mean_h  = _pad(entropy_histories).mean(axis=0)
    std_h   = _pad(entropy_histories).std(axis=0)
    mean_kl = _pad(kl_histories).mean(axis=0)
    std_kl  = _pad(kl_histories).std(axis=0)
    steps   = np.arange(max_len)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Entropy panel ─────────────────────────────────────────────────────────
    axes[0].plot(steps, mean_h, color="steelblue", linewidth=1.5)
    axes[0].fill_between(
        steps, mean_h - std_h, mean_h + std_h,
        alpha=0.25, color="steelblue", label="±1 std",
    )
    axes[0].set_xlabel("Step within episode")
    axes[0].set_ylabel("Mean entropy (bits)")
    axes[0].set_title(f"Belief entropy over time  (noise={noise_prob})")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linewidth=0.5, alpha=0.6)

    # ── KL divergence panel ───────────────────────────────────────────────────
    axes[1].plot(steps, mean_kl, color="coral", linewidth=1.5)
    axes[1].fill_between(
        steps, mean_kl - std_kl, mean_kl + std_kl,
        alpha=0.25, color="coral", label="±1 std",
    )
    axes[1].set_xlabel("Step within episode")
    axes[1].set_ylabel("KL divergence (bits)")
    axes[1].set_title(f"Belief vs ground truth KL  (noise={noise_prob})")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linewidth=0.5, alpha=0.6)

    plt.suptitle(
        f"Belief map — {len(entropy_histories)} episodes, noise_prob={noise_prob}",
        fontsize=11,
    )
    plt.tight_layout()

    out_path = Path(model_path).parent / f"belief_entropy_kl_noise{noise_prob}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved belief metrics plot to: {out_path}")


def _plot_path_optimality(ratios: list, noise_prob: float, model_path: str) -> None:
    """
    Save a bar chart of per-episode path optimality ratios next to the model.

    Each bar shows actual_cells / optimal_a_star_cells for one successful
    episode.  The dashed red line at 1.0 marks the theoretical optimum.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(ratios) * 0.4 + 2), 4))

    xs = np.arange(1, len(ratios) + 1)
    ax.bar(xs, ratios, color="steelblue", alpha=0.75, width=0.6)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Optimal (ratio = 1.0)")

    ax.set_xlabel("Episode (successful only)")
    ax.set_ylabel("Path optimality ratio")
    ax.set_title(
        f"Path Optimality Ratio  (noise={noise_prob})\n"
        f"mean={np.mean(ratios):.2f}  std={np.std(ratios):.2f}  "
        f"[{len(ratios)} episodes]"
    )
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    out_path = Path(model_path).parent / f"path_optimality_noise{noise_prob}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved path optimality plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--video", action="store_true", help="Save videos to disk")
    parser.add_argument("--video_dir", type=str, default="videos", help="Folder to save mp4 videos")
    parser.add_argument("--video_prefix", type=str, default="ppo_simplekeydoor", help="Video filename prefix")
    parser.add_argument("--noise", type=float, default=0.0, help="Sensor noise probability (0.0 = clean)")
    parser.add_argument("--grid_size", type=int, default=None, help="Grid size (default: read from config.json next to model)")
    parser.add_argument("--n_lava", type=int, default=None, help="Number of lava tiles (default: read from config.json next to model)")
    parser.add_argument("--belief", action="store_true",
                        help="Track belief map per step; save entropy/KL plot next to model")
    parser.add_argument("--only_success", action="store_true",
                        help="Only show/record successful episodes; runs until N successes collected")
    parser.add_argument("--fixed_seed", type=int, default=None,
                        help="Fix the map layout by using this seed on every episode reset (for fair model comparison)")

    args = parser.parse_args()

    # Auto-load grid_size / n_lava from the config.json saved alongside the model
    config_path = Path(args.model).parent / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if args.grid_size is None:
            args.grid_size = saved.get("grid_size", 13)
        if args.n_lava is None:
            args.n_lava = saved.get("n_lava", 0)
    else:
        if args.grid_size is None:
            args.grid_size = 13
        if args.n_lava is None:
            args.n_lava = 0

    register_env()

    def make_eval_env():

        # - If recording video, render_mode MUST be rgb_array
        # - Else keep your existing behavior
        render_mode = "rgb_array" if args.video else ("human" if args.render else None)

        e = gym.make("SimpleKeyDoor-v0", render_mode=render_mode, size=args.grid_size, n_lava=args.n_lava)
        e = ReducedActionSpace(e)
        if args.noise > 0.0:
            e = SensorNoiseWrapper(e, noise_prob=args.noise)
        e = ImgObsWrapper(e)

        if args.video:
            e = RecordVideo(
                e,
                video_folder=args.video_dir,
                episode_trigger=lambda ep_id: True,  # record every episode you run
                name_prefix=args.video_prefix,
                disable_logger=True,
            )
        return e

    env = DummyVecEnv([make_eval_env])
    env = VecTransposeImage(env)

    model = PPO.load(args.model, env=env, custom_objects={"features_extractor_class": MiniGridCNN})

    # Stable reference to the underlying MiniGrid env (survives resets).
    base = env.venv.envs[0].unwrapped

    # Belief map — created once and reset each episode.
    bmap = BeliefMap(grid_size=args.grid_size, noise_prob=args.noise) if args.belief else None
    all_entropy_histories: list = []   # one list of floats per episode
    all_kl_histories: list = []

    ep_rewards        = []
    ep_path_ratios    = []   # path optimality ratio for each successful episode
    successes         = 0
    ep_shown          = 0        # episodes printed/counted (target = args.episodes)
    ep_attempted      = 0        # total episodes run (used for real success-rate calc)
    max_attempts      = args.episodes * 20   # safety cap when --only_success

    while ep_shown < args.episodes:
        if args.only_success and ep_attempted >= max_attempts:
            print(f"Warning: hit attempt limit ({max_attempts}). "
                  f"Collected {ep_shown}/{args.episodes} successes.")
            break

        if args.fixed_seed is not None:
            # Force the same map layout every episode across models/noise levels.
            # env_method calls reset(seed=...) on the underlying gymnasium env,
            # which seeds MiniGrid's RNG before generating the grid.
            result = env.env_method("reset", seed=args.fixed_seed)
            obs_hwc = result[0][0]                               # (H, W, C)
            obs = obs_hwc[np.newaxis].transpose(0, 3, 1, 2)     # (1, C, H, W)
        else:
            obs = env.reset()

        done  = False
        total = 0.0
        step  = 0

        # ── A* path optimality: capture episode-start state ───────────────────
        # Read positions and grid NOW (before any steps change them).
        ep_start_pos  = tuple(base.agent_pos)
        ep_key_pos    = tuple(base.key_pos)
        ep_door_pos   = tuple(base.door_pos)
        ep_goal_pos   = tuple(base.goal_pos)
        opt_len = optimal_path_length(
            base.grid, args.grid_size,
            start=ep_start_pos,
            key_pos=ep_key_pos,
            door_pos=ep_door_pos,
            goal_pos=ep_goal_pos,
        )
        prev_agent_pos = ep_start_pos
        actual_cells   = 0   # counts successful forward moves (position changed)

        # ── Belief map: reset and initialise with the first observation ───────
        if bmap is not None:
            bmap.reset()
            bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)
            ep_entropy = [bmap.mean_entropy()]
            ep_kl      = [bmap.kl_from_truth(base.grid)]

        step_logs = []   # buffer per-step prints; flushed only if success (--only_success)

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "toggle"}
            a = int(action) if np.isscalar(action) else int(action[0])

            # ----- DEBUG BLOCK -----
            agent_pos = tuple(base.agent_pos)
            goal_pos  = tuple(base.goal_pos)

            door = base.grid.get(*base.door_pos)
            door_open   = door is not None and door.type == "door" and door.is_open
            door_locked = door is not None and door.type == "door" and door.is_locked

            carrying = None if base.carrying is None else base.carrying.type

            log_line = (
                f"pos={agent_pos} | goal={goal_pos} | "
                f"door_open={door_open} | door_locked={door_locked} | "
                f"carrying={carrying} | action={a}:{action_names.get(a)}"
            )
            if args.only_success:
                step_logs.append(log_line)   # hold until we know outcome
            else:
                print(log_line)
            # ----- END DEBUG BLOCK -----

            obs, reward, dones, infos = env.step(action)
            total += float(reward[0])
            done = bool(dones[0])
            step += 1

            # ── Track actual cell moves for path optimality ───────────────────
            new_pos = tuple(base.agent_pos)
            if new_pos != prev_agent_pos:
                actual_cells += 1
            prev_agent_pos = new_pos

            # ── Belief map update with the new observation ────────────────────
            if bmap is not None:
                bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)
                ep_entropy.append(bmap.mean_entropy())
                ep_kl.append(bmap.kl_from_truth(base.grid))

        ep_attempted += 1
        is_success = infos[0].get("is_success", False)

        # In --only_success mode, discard failed episodes (print brief status)
        if args.only_success and not is_success:
            print(f"  [attempt {ep_attempted}] failed ({step} steps) — skipping")
            continue

        # Flush buffered step logs for episodes we're showing
        if args.only_success:
            for line in step_logs:
                print(line)

        ep_shown += 1
        print(f"Episode {ep_shown} finished in {step} steps")
        final_pos = tuple(infos[0].get("agent_pos"))
        goal_pos  = tuple(infos[0].get("goal_pos"))
        print(f"FINAL POS (from info): {final_pos} | GOAL POS: {goal_pos}")
        print("SUCCESS:", is_success)

        # ── Path optimality ratio (successful episodes only) ──────────────────
        if is_success and opt_len is not None and opt_len > 0:
            ratio = actual_cells / opt_len
            ep_path_ratios.append(ratio)
            print(
                f"Path optimality: {actual_cells} cells moved / "
                f"{opt_len} A* optimal = {ratio:.2f}x  "
                f"(start={ep_start_pos} key={ep_key_pos} "
                f"door={ep_door_pos} goal={ep_goal_pos})"
            )

        ep_rewards.append(total)
        if bmap is not None:
            all_entropy_histories.append(ep_entropy)
            all_kl_histories.append(ep_kl)

        if is_success:
            successes += 1

    env.close()

    if bmap is not None and all_entropy_histories:
        _plot_belief_metrics(all_entropy_histories, all_kl_histories, args.noise, args.model)

    if ep_path_ratios:
        _plot_path_optimality(ep_path_ratios, args.noise, args.model)

    if args.video:
        print(f"✅ Videos saved to: {args.video_dir}")

    print(f"\nEpisodes shown: {ep_shown}")
    if args.only_success:
        print(f"Success rate: {successes}/{ep_attempted} attempts = {successes/ep_attempted:.2%}")
    else:
        print(f"Success rate: {successes / ep_attempted:.2%}")
    print(f"Mean reward: {np.mean(ep_rewards):.3f} | Std: {np.std(ep_rewards):.3f}")

    if ep_path_ratios:
        print(
            f"Path optimality ratio: mean={np.mean(ep_path_ratios):.2f} "
            f"std={np.std(ep_path_ratios):.2f} "
            f"(1.0 = optimal, higher = more detours) "
            f"[{len(ep_path_ratios)}/{ep_shown} episodes had a valid A* reference]"
        )


if __name__ == "__main__":
    main()