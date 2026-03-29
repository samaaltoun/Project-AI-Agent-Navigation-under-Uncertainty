# eval_ppo_var.py
# This file is built on the initial eval_ppo but for evaluating a PPO agent trained on the newer VariationalGoalEnv
# The idea is that different parameters can be used to make the grid different and more challenging for the agent
# The SimpleKeyDoor environment is too easily solved by the agent so we added differing dimensions, multiple walls, 
# and different wall orientations to see how the agent would respond

import argparse
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import ImgObsWrapper, ViewSizeWrapper
from gymnasium.wrappers import RecordVideo

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.variational_goal_env import LevelConfig, VariationalGoalEnv, register_env
from envs.wrappers import ReducedActionSpace


class MiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str,   required=True)
    parser.add_argument("--episodes",     type=int,   default=20)
    parser.add_argument("--render",       action="store_true")
    parser.add_argument("--video",        action="store_true")
    parser.add_argument("--video_dir",    type=str,   default="videos")
    parser.add_argument("--video_prefix", type=str,   default="ppo_variationalgoal")
    parser.add_argument("--view_size", type=int, default=11)

    # LevelConfig args — must match what the model was trained on
    parser.add_argument("--width",            type=int,   default=20)
    parser.add_argument("--height",           type=int,   default=16)
    parser.add_argument("--num_walls",        type=int,   default=1)
    parser.add_argument("--doors_per_wall",   type=int,   default=1)
    parser.add_argument("--wall_orientation", type=str,   default="vertical",
                        choices=["vertical", "horizontal", "random"])
    parser.add_argument("--min_gap",          type=int,   default=3)

    args = parser.parse_args()

    register_env()

    level = LevelConfig(
        width=args.width,
        height=args.height,
        num_walls=args.num_walls,
        doors_per_wall=args.doors_per_wall,
        wall_orientation=args.wall_orientation,
        min_gap=args.min_gap,
    )

    def make_eval_env():
        render_mode = "rgb_array" if args.video else ("human" if args.render else None)
        env = VariationalGoalEnv(level=level, render_mode=render_mode)
        env = ReducedActionSpace(env)
        env = ViewSizeWrapper(env, agent_view_size=args.view_size)
        env = ImgObsWrapper(env)
        if args.video:
            env = RecordVideo(
                env,
                video_folder=args.video_dir,
                episode_trigger=lambda ep_id: True,
                name_prefix=args.video_prefix,
                disable_logger=True,
            )
        return env

    env = DummyVecEnv([make_eval_env])
    env = VecTransposeImage(env)
    model = PPO.load(args.model, custom_objects={})

    action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "toggle"}
    successes = 0
    ep_rewards = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            a = int(action) if np.isscalar(action) else int(action[0])

            # --- debug ---
            base = env.venv.envs[0].unwrapped
            agent_pos = tuple(base.agent_pos)
            carrying = None if base.carrying is None else base.carrying.type
            doors_info = [
                f"door{i}@{dp}={'open' if base._door_is_open(dp) else 'locked'}"
                for i, dp in enumerate(base.door_positions)
            ]
            print(
                f"pos={agent_pos} | {' '.join(doors_info)} | "
                f"keys_picked={base._keys_picked} | "
                f"carrying={carrying} | action={a}:{action_names.get(a)}"
            )
            # --- end debug ---

            obs, reward, dones, infos = env.step(action)
            total += float(reward[0])
            done = bool(dones[0])
            steps += 1

        ep_rewards.append(total)
        success = infos[0].get("is_success", False)
        if success:
            successes += 1

        print(
            f"Episode {ep + 1}: steps={steps} | reward={total:.3f} | "
            f"goal={infos[0].get('goal_pos')} | success={success}"
        )

    env.close()

    if args.video:
        print(f"Videos saved to: {args.video_dir}")

    print(f"\nEpisodes:     {args.episodes}")
    print(f"Success rate: {successes / args.episodes:.2%}")
    print(f"Mean reward:  {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}")


if __name__ == "__main__":
    main()
