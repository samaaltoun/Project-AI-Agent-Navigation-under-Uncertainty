# eval_ppo_var.py
import argparse
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import RecordVideo

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.variational_goal_env import VariationalGoalEnv, register_env
from envs.wrappers import ReducedActionSpace


class MiniGridCNN(BaseFeaturesExtractor):
    """CNN feature extractor for MiniGrid observations (channel-first after VecTransposeImage)."""

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
    parser.add_argument("--model", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", action="store_true", help="Save videos to disk")
    parser.add_argument("--video_dir", type=str, default="videos")
    parser.add_argument("--video_prefix", type=str, default="ppo_variationalgoal")
    parser.add_argument("--grid_size", type=int, default=10)
    args = parser.parse_args()

    register_env()

    def make_eval_env():
        render_mode = "rgb_array" if args.video else ("human" if args.render else None)
        env = gym.make("VariationalGoal-v0", grid_size=args.grid_size, render_mode=render_mode)
        env = ReducedActionSpace(env)
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

            # ----- DEBUG BLOCK -----
            base = env.venv.envs[0].unwrapped

            agent_pos = tuple(base.agent_pos)
            goal_pos = tuple(base.goal_pos)

            door = base.grid.get(*base.door_pos)
            door_is_open = door is not None and door.type == "door" and door.is_open
            door_is_locked = door is not None and door.type == "door" and door.is_locked

            carrying = None if base.carrying is None else base.carrying.type

            print(
                f"pos={agent_pos} | goal={goal_pos} | "
                f"door_open={door_is_open} | door_locked={door_is_locked} | "
                f"carrying={carrying} | action={a}:{action_names.get(a)}"
            )
            # ----- END DEBUG BLOCK -----

            obs, reward, dones, infos = env.step(action)
            total += float(reward[0])
            done = bool(dones[0])
            steps += 1

        ep_rewards.append(total)

        final_pos = tuple(infos[0].get("agent_pos", (-1, -1)))
        goal_pos = tuple(infos[0].get("goal_pos", (-1, -1)))
        success = infos[0].get("is_success", False)

        print(f"Episode {ep + 1} finished in {steps} steps | reward={total:.3f}")
        print(f"  FINAL POS: {final_pos} | GOAL POS: {goal_pos} | SUCCESS: {success}")

        if success:
            successes += 1

    env.close()

    if args.video:
        print(f"Videos saved to: {args.video_dir}")

    print(f"\nEpisodes:      {args.episodes}")
    print(f"Success rate:  {successes / args.episodes:.2%}")
    print(f"Mean reward:   {np.mean(ep_rewards):.3f} ± {np.std(ep_rewards):.3f}")


if __name__ == "__main__":
    main()
