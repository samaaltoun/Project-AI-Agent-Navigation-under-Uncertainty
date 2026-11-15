"""
noise_sweep.py: evaluate a trained PPO model at multiple sensor-noise levels
and produce a three-panel comparison plot.

What it measures
For each noise level in --noise_levels (default: 0.0, 0.1, 0.2, 0.5, 1.0):

  • success_rate: fraction of episodes where the agent reached the goal
  • mean_kl: mean KL divergence (belief ‖ ground truth), averaged
                     over every step of every episode
  • mean_entropy: mean belief-map entropy, averaged the same way

Output
  <run_dir>/noise_sweep.png  — three-panel plot saved next to model.zip

Usage
  python noise_sweep.py --model runs/2026-03-28_.../model.zip --episodes 50
  python noise_sweep.py --model ... --episodes 100 --noise_levels 0.0 0.2 0.5
"""

from __future__ import annotations
import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")   #save-only backend, works without a display
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import ImgObsWrapper
from envs.simple_keydoor_env import register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper
from envs.extractors import MiniGridCNN
from envs.belief_map import BeliefMap


#env factoryy
def _make_env(noise_prob: float, grid_size: int, n_lava: int = 0):
    e = gym.make("SimpleKeyDoor-v0", render_mode=None, size=grid_size, n_lava=n_lava)
    e = ReducedActionSpace(e)
    if noise_prob > 0.0:
        e = SensorNoiseWrapper(e, noise_prob=noise_prob)
    e = ImgObsWrapper(e)
    return e

#per noise level eval
def _run_level(
    model_path: str,
    grid_size: int,
    noise_prob: float,
    n_episodes: int,
    n_lava: int = 0,
) -> tuple[float, float, float]:
    """
    Load model fresh with a 1-env at noise_prob, run n_episodes, return:
        (success_rate, mean_kl_per_step, mean_entropy_per_step)
    """
    #Build a fresh 1-env for this noise level.
    vec_env = DummyVecEnv([lambda: _make_env(noise_prob, grid_size, n_lava)])
    vec_env = VecTransposeImage(vec_env)
    # Load fresh so avoids n_envs mismatch from set_env (model was trained with 8 envs).
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

    for _ in range(n_episodes):
        obs = vec_env.reset()
        done = False

        bmap.reset()
        bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)

        step_kl      = [bmap.kl_from_truth(base.grid)]
        step_entropy = [bmap.mean_entropy()]

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _reward, dones, infos = vec_env.step(action)
            done = bool(dones[0])

            bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)
            step_kl.append(bmap.kl_from_truth(base.grid))
            step_entropy.append(bmap.mean_entropy())

        if infos[0].get("is_success", False):
            successes += 1

        ep_mean_kls.append(float(np.mean(step_kl)))
        ep_mean_entropies.append(float(np.mean(step_entropy)))

    vec_env.close()

    return (
        successes / n_episodes,
        float(np.mean(ep_mean_kls)),
        float(np.mean(ep_mean_entropies)),
    )


#plotting

def _save_plot(
    noise_levels: list[float],
    success_rates: list[float],
    mean_kls: list[float],
    mean_entropies: list[float],
    model_path: str,
    n_episodes: int,
) -> None:
    """Three-panel figure: success rate, KL divergence, entropy — all vs noise."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = noise_levels

    #panel 1 for success rate
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

    #panel 2 kl diverg
    axes[1].plot(xs, mean_kls, "o-", color="coral", linewidth=2, markersize=7)
    axes[1].set_xlabel("Noise probability")
    axes[1].set_ylabel("Mean KL divergence (bits)")
    axes[1].set_title("Belief KL divergence vs noise level")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xticks(xs)
    axes[1].grid(True, linewidth=0.5, alpha=0.6)

    #panel 3 entropy
    axes[2].plot(xs, mean_entropies, "o-", color="seagreen", linewidth=2, markersize=7)
    axes[2].set_xlabel("Noise probability")
    axes[2].set_ylabel("Mean entropy (bits)")
    axes[2].set_title("Belief entropy vs noise level")
    axes[2].set_ylim(bottom=0)
    axes[2].set_xticks(xs)
    axes[2].grid(True, linewidth=0.5, alpha=0.6)

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


#main

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

    noise_levels = sorted(args.noise_levels)
    success_rates:   list[float] = []
    mean_kls:        list[float] = []
    mean_entropies:  list[float] = []

    for noise in noise_levels:
        print(f"\n── noise={noise:.2f}  ({args.episodes} episodes) ──")
        sr, mkl, mh = _run_level(args.model, args.grid_size, noise, args.episodes, args.n_lava)
        success_rates.append(sr)
        mean_kls.append(mkl)
        mean_entropies.append(mh)
        print(f"   success={sr:.2%}   mean_kl={mkl:.3f} bits   mean_entropy={mh:.3f} bits")

    print("\n── Summary ─────────────────────────────────────────────────────────")
    print(f"{'noise':>6}  {'success':>8}  {'mean_kl':>9}  {'mean_H':>8}")
    for noise, sr, mkl, mh in zip(noise_levels, success_rates, mean_kls, mean_entropies):
        print(f"{noise:6.2f}  {sr:8.2%}  {mkl:9.3f}  {mh:8.3f}")

    _save_plot(noise_levels, success_rates, mean_kls, mean_entropies, args.model, args.episodes)


if __name__ == "__main__":
    main()
