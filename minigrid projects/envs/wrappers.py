import gymnasium as gym
from gymnasium import spaces

class ReducedActionSpace(gym.Wrapper):
    # keep: left, right, forward, pickup, toggle
    def __init__(self, env):
        super().__init__(env)
        self.keep = [0, 1, 2, 3, 5]
        self.action_space = spaces.Discrete(len(self.keep))

    def step(self, action):
        return self.env.step(self.keep[int(action)])