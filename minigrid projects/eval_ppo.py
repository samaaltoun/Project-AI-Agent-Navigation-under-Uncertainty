import argparse
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import ImgObsWrapper

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

#from envs.simple_goal_env import register_env
from envs.simple_keydoor_env import register_env
from envs.wrappers import ReducedActionSpace

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    register_env()

    def make_eval_env():
        e = gym.make("SimpleKeyDoor-v0", render_mode="human" if args.render else None)
        e = ReducedActionSpace(e)
        e = ImgObsWrapper(e)
        return e

    env = DummyVecEnv([make_eval_env])
    env = VecTransposeImage(env)

    model = PPO.load(args.model, custom_objects={})

    ep_rewards = []
    successes = 0

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total = 0.0
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_names = {0:"left", 1:"right", 2:"forward", 3:"pickup", 4:"toggle"}
            a = int(action) if np.isscalar(action) else int(action[0])

            # ----- DEBUG BLOCK -----
            base = env.venv.envs[0].unwrapped  # access underlying MiniGrid env

            agent_pos = tuple(base.agent_pos)
            goal_pos = tuple(base.goal_pos)

            door = base.grid.get(*base.door_pos)
            door_is_open = (door is not None and door.type == "door" and door.is_open)
            door_is_locked = (door is not None and door.type == "door" and door.is_locked)

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
            step += 1
        print(f"Episode {ep + 1} finished in {step} steps")
        if done:
            final_pos = tuple(infos[0].get("agent_pos"))
            goal_pos = tuple(infos[0].get("goal_pos"))
            print(f"FINAL POS (from info): {final_pos} | GOAL POS: {goal_pos}")
            print("SUCCESS:", infos[0].get("is_success", False))

        ep_rewards.append(total)

        # Simple success proxy: in your env, reaching goal should give noticeably positive total reward.
        # You can refine later (e.g., check env.unwrapped.agent_pos == goal_pos).
        if infos[0].get("is_success", False):
            successes += 1

    env.close()

    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {successes / args.episodes:.2%}")
    print(f"Mean reward: {np.mean(ep_rewards):.3f} | Std: {np.std(ep_rewards):.3f}")


if __name__ == "__main__":
    main()