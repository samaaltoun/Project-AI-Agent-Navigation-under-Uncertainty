"""
eval_random.py — evaluate a random-action baseline on SimpleKeyDoor.

Runs N episodes where the agent picks a uniformly random action each step.
Reports the same metrics as eval_ppo.py: success rate, mean reward,
and path optimality ratio.

Usage
-----
  python eval_random.py --episodes 100 --grid_size 15
  python eval_random.py --episodes 50 --noise 0.3 --n_lava 5
"""

from __future__ import annotations

import argparse
import numpy as np
import gymnasium as gym

from envs.simple_keydoor_env import register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper
from envs.pathfinding import optimal_path_length
from minigrid.wrappers import ImgObsWrapper


def main():
    parser = argparse.ArgumentParser(description="Random baseline for SimpleKeyDoor")
    parser.add_argument("--episodes",  type=int,   default=100)
    parser.add_argument("--grid_size", type=int,   default=15)
    parser.add_argument("--n_lava",    type=int,   default=0)
    parser.add_argument("--noise",     type=float, default=0.0,
                        help="Sensor noise probability (for fair comparison with PPO eval)")
    args = parser.parse_args()

    register_env()

    ep_rewards = []
    ep_optimality = []
    successes = 0

    for ep in range(args.episodes):
        env = gym.make("SimpleKeyDoor-v0", render_mode=None,
                       size=args.grid_size, n_lava=args.n_lava)
        env = ReducedActionSpace(env)
        if args.noise > 0.0:
            env = SensorNoiseWrapper(env, noise_prob=args.noise)
        env = ImgObsWrapper(env)

        obs, info = env.reset()
        base = env.unwrapped

        opt_len = optimal_path_length(
            base.grid,
            tuple(base.agent_pos),
            tuple(base.key_pos),
            tuple(base.door_pos),
            tuple(base.goal_pos),
        )

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

        is_success = info.get("is_success", False)
        if is_success:
            successes += 1

        ep_rewards.append(total_reward)

        if opt_len > 0 and steps > 0:
            ratio = opt_len / steps
            ep_optimality.append(ratio)

        status = "SUCCESS" if is_success else "FAIL"
        print(f"Episode {ep + 1:>4}  steps={steps:>4}  reward={total_reward:>7.2f}  {status}")
        env.close()

    print(f"\n── Random Baseline Summary ─────────────────────────────────────────")
    print(f"Episodes:    {args.episodes}")
    print(f"Grid size:   {args.grid_size}")
    print(f"Noise:       {args.noise}")
    print(f"Success rate: {successes}/{args.episodes} = {successes / args.episodes:.2%}")
    print(f"Mean reward:  {np.mean(ep_rewards):.3f} | Std: {np.std(ep_rewards):.3f}")
    if ep_optimality:
        print(f"Mean path optimality: {np.mean(ep_optimality):.3f} | Std: {np.std(ep_optimality):.3f}")
    else:
        print("Mean path optimality: N/A (no successful episodes)")


if __name__ == "__main__":
    main()
