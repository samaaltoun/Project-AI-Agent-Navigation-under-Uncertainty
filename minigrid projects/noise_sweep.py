"""
noise_sweep.py — evaluate a trained PPO model at multiple sensor-noise levels
and produce a three-panel comparison plot.

What it measures
----------------
For each noise level in --noise_levels (default: 0.0, 0.1, 0.2, 0.5, 1.0):

  • success_rate   — fraction of episodes where the agent reached the goal
  • mean_kl        — mean KL divergence (belief ‖ ground truth), averaged
                     over every step of every episode
  • mean_entropy   — mean belief-map entropy, averaged the same way

Output
------
  <run_dir>/noise_sweep.png  — three-panel plot saved next to model.zip

Usage
-----
  python noise_sweep.py --model runs/2026-03-28_.../model.zip --episodes 50
  python noise_sweep.py --model ... --episodes 100 --noise_levels 0.0 0.2 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

import matplotlib
matplotlib.use("Agg")   # save-only backend, works without a display
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import ImgObsWrapper

from envs.simple_keydoor_env import register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper
from envs.extractors import MiniGridCNN
from envs.belief_map import BeliefMap
from envs.astar import optimal_path_length


# ── Environment factory ────────────────────────────────────────────────────────

def _make_env(noise_prob: float, grid_size: int, n_lava: int = 0):
    e = gym.make("SimpleKeyDoor-v0", render_mode=None, size=grid_size, n_lava=n_lava)
    e = ReducedActionSpace(e)
    if noise_prob > 0.0:
        e = SensorNoiseWrapper(e, noise_prob=noise_prob)
    e = ImgObsWrapper(e)
    return e


# ── Per-noise-level evaluation ─────────────────────────────────────────────────

def _run_level(
    model_path: str,
    grid_size: int,
    noise_prob: float,
    n_episodes: int,
    n_lava: int = 0,
) -> tuple[float, float, float, float | None]:
    """
    Load model fresh with a 1-env at noise_prob, run n_episodes, return:
        (success_rate, mean_kl_per_step, mean_entropy_per_step, mean_path_optimality_ratio)

    mean_path_optimality_ratio is None if no successful episodes occurred.
    """
    # Build a fresh 1-env for this noise level.
    vec_env = DummyVecEnv([lambda: _make_env(noise_prob, grid_size, n_lava)])
    vec_env = VecTransposeImage(vec_env)
    # Load fresh — avoids n_envs mismatch from set_env (model was trained with 8 envs).
    model = PPO.load(
        model_path,
        env=vec_env,
        custom_objects={"features_extractor_class": MiniGridCNN},
    )

    base = vec_env.venv.envs[0].unwrapped
    bmap = BeliefMap(grid_size=grid_size, noise_prob=noise_prob)

    successes = 0
    ep_mean_kls: list[float] = []
    ep_mean_entropies: list[float] = []
    ep_path_ratios: list[float] = []

    for _ in range(n_episodes):
        obs = vec_env.reset()
        done = False

        # ── Capture initial state for A* (must happen right after reset) ──────
        ep_start_pos = tuple(base.agent_pos)
        ep_key_pos   = tuple(base.key_pos)
        ep_door_pos  = tuple(base.door_pos)
        ep_goal_pos  = tuple(base.goal_pos)
        opt_len = optimal_path_length(
            base.grid, grid_size,
            start=ep_start_pos,
            key_pos=ep_key_pos,
            door_pos=ep_door_pos,
            goal_pos=ep_goal_pos,
        )
        prev_pos     = ep_start_pos
        actual_cells = 0

        bmap.reset()
        bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)

        step_kl      = [bmap.kl_from_truth(base.grid)]
        step_entropy = [bmap.mean_entropy()]

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _reward, dones, infos = vec_env.step(action)
            done = bool(dones[0])

            # Track cell moves
            new_pos = tuple(base.agent_pos)
            if new_pos != prev_pos:
                actual_cells += 1
            prev_pos = new_pos

            bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)
            step_kl.append(bmap.kl_from_truth(base.grid))
            step_entropy.append(bmap.mean_entropy())

        is_success = infos[0].get("is_success", False)
        if is_success:
            successes += 1
            if opt_len is not None and opt_len > 0:
                ep_path_ratios.append(actual_cells / opt_len)

        ep_mean_kls.append(float(np.mean(step_kl)))
        ep_mean_entropies.append(float(np.mean(step_entropy)))

    vec_env.close()

    mean_ratio = float(np.mean(ep_path_ratios)) if ep_path_ratios else None
    return (
        successes / n_episodes,
        float(np.mean(ep_mean_kls)),
        float(np.mean(ep_mean_entropies)),
        mean_ratio,
    )


# ── Plotting ───────────────────────────────────────────────────────────────────

def _save_plot(
    noise_levels: list[float],
    success_rates: list[float],
    mean_kls: list[float],
    mean_entropies: list[float],
    mean_path_ratios: list[float | None],
    model_path: str,
    n_episodes: int,
) -> None:
    """Four-panel figure: success rate, KL divergence, entropy, path optimality — all vs noise."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    xs = noise_levels

    # ── Panel 1: Success rate ──────────────────────────────────────────────────
    axes[0].plot(xs, success_rates, "o-", color="steelblue", linewidth=2, markersize=7)
    for x, y in zip(xs, success_rates):
        axes[0].annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[0].set_xlabel("Noise probability")
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Success rate vs noise level")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_xticks(xs)
    axes[0].grid(True, linewidth=0.5, alpha=0.6)

    # ── Panel 2: KL divergence ─────────────────────────────────────────────────
    axes[1].plot(xs, mean_kls, "o-", color="coral", linewidth=2, markersize=7)
    axes[1].set_xlabel("Noise probability")
    axes[1].set_ylabel("Mean KL divergence (bits)")
    axes[1].set_title("Belief KL divergence vs noise level")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xticks(xs)
    axes[1].grid(True, linewidth=0.5, alpha=0.6)

    # ── Panel 3: Entropy ───────────────────────────────────────────────────────
    axes[2].plot(xs, mean_entropies, "o-", color="seagreen", linewidth=2, markersize=7)
    axes[2].set_xlabel("Noise probability")
    axes[2].set_ylabel("Mean entropy (bits)")
    axes[2].set_title("Belief entropy vs noise level")
    axes[2].set_ylim(bottom=0)
    axes[2].set_xticks(xs)
    axes[2].grid(True, linewidth=0.5, alpha=0.6)

    # ── Panel 4: Path optimality ratio ────────────────────────────────────────
    # Only plot noise levels where we had at least one successful episode.
    valid_xs     = [x for x, r in zip(xs, mean_path_ratios) if r is not None]
    valid_ratios = [r for r in mean_path_ratios if r is not None]
    if valid_xs:
        axes[3].plot(valid_xs, valid_ratios, "o-", color="mediumpurple",
                     linewidth=2, markersize=7)
        axes[3].axhline(1.0, color="red", linestyle="--", linewidth=1.2,
                        label="Optimal (1.0)")
        for x, r in zip(valid_xs, valid_ratios):
            axes[3].annotate(f"{r:.2f}", (x, r), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=8)
        axes[3].legend(fontsize=8)
    else:
        axes[3].text(0.5, 0.5, "No successful episodes", ha="center",
                     va="center", transform=axes[3].transAxes, fontsize=10)
    axes[3].set_xlabel("Noise probability")
    axes[3].set_ylabel("Mean path optimality ratio")
    axes[3].set_title("Path optimality ratio vs noise level\n(successful episodes only)")
    axes[3].set_ylim(bottom=0)
    axes[3].set_xticks(xs)
    axes[3].grid(True, linewidth=0.5, alpha=0.6)

    plt.suptitle(
        f"Noise sweep  —  {n_episodes} episodes per level  "
        f"(model: {Path(model_path).parent.name})",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = Path(model_path).parent / "noise_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved noise sweep plot to: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Noise-level sweep for PPO + BeliefMap")
    parser.add_argument("--model",    type=str, required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=50,    help="Episodes per noise level")
    parser.add_argument("--grid_size",  type=int, default=13, help="Grid size (must match training)")
    parser.add_argument("--n_lava",type=int, default=0,  help="Number of wall obstacles (must match training)")
    parser.add_argument(
        "--noise_levels", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.5, 1.0],
        help="Noise probabilities to sweep (space-separated floats)",
    )
    args = parser.parse_args()

    register_env()

    noise_levels      = sorted(args.noise_levels)
    success_rates:    list[float]        = []
    mean_kls:         list[float]        = []
    mean_entropies:   list[float]        = []
    mean_path_ratios: list[float | None] = []

    for noise in noise_levels:
        print(f"\n── noise={noise:.2f}  ({args.episodes} episodes) ──")
        sr, mkl, mh, mpr = _run_level(args.model, args.grid_size, noise, args.episodes, args.n_lava)
        success_rates.append(sr)
        mean_kls.append(mkl)
        mean_entropies.append(mh)
        mean_path_ratios.append(mpr)
        ratio_str = f"{mpr:.2f}x" if mpr is not None else "n/a"
        print(f"   success={sr:.2%}   mean_kl={mkl:.3f} bits   mean_entropy={mh:.3f} bits   path_ratio={ratio_str}")

    print("\n── Summary ─────────────────────────────────────────────────────────")
    print(f"{'noise':>6}  {'success':>8}  {'mean_kl':>9}  {'mean_H':>8}  {'path_ratio':>11}")
    for noise, sr, mkl, mh, mpr in zip(noise_levels, success_rates, mean_kls, mean_entropies, mean_path_ratios):
        ratio_str = f"{mpr:.2f}x" if mpr is not None else "       n/a"
        print(f"{noise:6.2f}  {sr:8.2%}  {mkl:9.3f}  {mh:8.3f}  {ratio_str:>11}")

    _save_plot(noise_levels, success_rates, mean_kls, mean_entropies, mean_path_ratios, args.model, args.episodes)


if __name__ == "__main__":
    main()
