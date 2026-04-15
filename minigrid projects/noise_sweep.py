"""
noise_sweep.py — evaluate a trained PPO model at multiple sensor-noise levels
and produce a four-panel comparison plot.

Usage
-----
  python noise_sweep.py --model runs/<folder>/model.zip --episodes 50
  python noise_sweep.py --model ... --episodes 100 --noise_levels 0.0 0.2 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import ImgObsWrapper

from envs.simple_keydoor_env import register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper, BeliefObsWrapper
from envs.extractors import MiniGridCNN
from envs.belief_map import BeliefMap
from envs.pathfinding import optimal_path_length


# ── Environment factory ────────────────────────────────────────────────────────

def _make_env(noise_prob: float, grid_size: int, n_lava: int = 0,
              use_belief_obs: bool = False, tpr=None, decay: float = 0.0):
    e = gym.make("SimpleKeyDoor-v0", render_mode=None, size=grid_size, n_lava=n_lava)
    e = ReducedActionSpace(e)
    if noise_prob > 0.0:
        e = SensorNoiseWrapper(e, noise_prob=noise_prob)
    if use_belief_obs:
        e = BeliefObsWrapper(e, grid_size=grid_size, noise_prob=noise_prob,
                             tpr=tpr, decay=decay)
    else:
        e = ImgObsWrapper(e)
    return e


# ── Per-noise-level evaluation ─────────────────────────────────────────────────

def _run_level(
    model_path: str,
    grid_size: int,
    noise_prob: float,
    n_episodes: int,
    n_lava: int = 0,
    use_belief_obs: bool = False,
    tpr=None,
    decay: float = 0.0,
) -> tuple[float, float, float, float]:
    vec_env = DummyVecEnv([lambda: _make_env(
        noise_prob, grid_size, n_lava, use_belief_obs, tpr, decay
    )])
    if not use_belief_obs:
        vec_env = VecTransposeImage(vec_env)  # BeliefObsWrapper already outputs (C,H,W)
    model = PPO.load(model_path, env=vec_env,
                     custom_objects={"features_extractor_class": MiniGridCNN})

    base = vec_env.venv.envs[0].unwrapped

    # Get belief map — either from wrapper or standalone
    belief_wrapper = None
    if use_belief_obs:
        w = vec_env.venv.envs[0]
        while w is not None:
            if isinstance(w, BeliefObsWrapper):
                belief_wrapper = w
                break
            w = getattr(w, "env", None)

    bmap = belief_wrapper.bmap if belief_wrapper else BeliefMap(
        grid_size=grid_size, noise_prob=noise_prob
    )

    successes = 0
    ep_mean_kls: list[float] = []
    ep_mean_entropies: list[float] = []
    ep_optimality: list[float] = []

    for _ in range(n_episodes):
        obs = vec_env.reset()
        done = False
        steps = 0

        opt_len = optimal_path_length(
            base.grid, tuple(base.agent_pos), tuple(base.key_pos),
            tuple(base.door_pos), tuple(base.goal_pos),
        )

        if belief_wrapper is None:
            bmap.reset()
            bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)

        step_kl      = [bmap.kl_from_truth(base.grid)]
        step_entropy = [bmap.mean_entropy()]

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _reward, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            steps += 1

            if belief_wrapper is None:
                bmap.update(obs[0].transpose(1, 2, 0), base.agent_pos, base.agent_dir)

            step_kl.append(bmap.kl_from_truth(base.grid))
            step_entropy.append(bmap.mean_entropy())

        if infos[0].get("is_success", False):
            successes += 1

        if opt_len > 0 and steps > 0:
            ep_optimality.append(opt_len / steps)

        ep_mean_kls.append(float(np.mean(step_kl)))
        ep_mean_entropies.append(float(np.mean(step_entropy)))

    vec_env.close()

    mean_opt = float(np.mean(ep_optimality)) if ep_optimality else 0.0

    return (
        successes / n_episodes,
        float(np.mean(ep_mean_kls)),
        float(np.mean(ep_mean_entropies)),
        mean_opt,
    )


# ── Plotting ───────────────────────────────────────────────────────────────────

def _save_plot(noise_levels, success_rates, mean_kls, mean_entropies,
               mean_optimalities, model_path, n_episodes):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    xs = noise_levels

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

    axes[1].plot(xs, mean_kls, "o-", color="coral", linewidth=2, markersize=7)
    axes[1].set_xlabel("Noise probability")
    axes[1].set_ylabel("Mean KL divergence (bits)")
    axes[1].set_title("Belief KL divergence vs noise level")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xticks(xs)
    axes[1].grid(True, linewidth=0.5, alpha=0.6)

    axes[2].plot(xs, mean_entropies, "o-", color="seagreen", linewidth=2, markersize=7)
    axes[2].set_xlabel("Noise probability")
    axes[2].set_ylabel("Mean entropy (bits)")
    axes[2].set_title("Belief entropy vs noise level")
    axes[2].set_ylim(bottom=0)
    axes[2].set_xticks(xs)
    axes[2].grid(True, linewidth=0.5, alpha=0.6)

    axes[3].plot(xs, mean_optimalities, "o-", color="darkorchid", linewidth=2, markersize=7)
    for x, y in zip(xs, mean_optimalities):
        axes[3].annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    axes[3].set_xlabel("Noise probability")
    axes[3].set_ylabel("Path optimality ratio")
    axes[3].set_title("Path optimality vs noise level")
    axes[3].set_ylim(0, 1.12)
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
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--grid_size",  type=int, default=None)
    parser.add_argument("--n_lava", type=int, default=None)
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.5, 1.0])
    args = parser.parse_args()

    # Auto-load config
    config_path = Path(args.model).parent / "config.json"
    saved = {}
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))

    if args.grid_size is None:
        args.grid_size = saved.get("grid_size", 15)
    if args.n_lava is None:
        args.n_lava = saved.get("n_lava", 0)

    use_belief_obs = "decay" in saved
    decay = saved.get("decay", 0.0)
    tpr = saved.get("tpr", None)

    register_env()

    noise_levels = sorted(args.noise_levels)
    success_rates:      list[float] = []
    mean_kls:           list[float] = []
    mean_entropies:     list[float] = []
    mean_optimalities:  list[float] = []

    for noise in noise_levels:
        print(f"\n── noise={noise:.2f}  ({args.episodes} episodes) ──")
        sr, mkl, mh, mopt = _run_level(
            args.model, args.grid_size, noise, args.episodes, args.n_lava,
            use_belief_obs=use_belief_obs, tpr=tpr, decay=decay,
        )
        success_rates.append(sr)
        mean_kls.append(mkl)
        mean_entropies.append(mh)
        mean_optimalities.append(mopt)
        print(f"   success={sr:.2%}   mean_kl={mkl:.3f} bits   mean_entropy={mh:.3f} bits   optimality={mopt:.3f}")

    print("\n── Summary ─────────────────────────────────────────────────────────")
    print(f"{'noise':>6}  {'success':>8}  {'mean_kl':>9}  {'mean_H':>8}  {'optim':>7}")
    for noise, sr, mkl, mh, mopt in zip(noise_levels, success_rates, mean_kls, mean_entropies, mean_optimalities):
        print(f"{noise:6.2f}  {sr:8.2%}  {mkl:9.3f}  {mh:8.3f}  {mopt:7.3f}")

    _save_plot(noise_levels, success_rates, mean_kls, mean_entropies, mean_optimalities, args.model, args.episodes)


if __name__ == "__main__":
    main()
