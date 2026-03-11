import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SensorNoiseWrapper(gym.ObservationWrapper):
    """
    Randomly corrupts visible cells in the agent's partial observation.

    noise_prob=0.0  -> no noise (clean observation)
    noise_prob=1.0  -> every visible cell is randomised

    Only visible cells (type != 0 / "unseen") are corrupted, so the
    distinction between "I can see something (but it may be wrong)" and
    "I cannot see this area at all" is preserved.
    """

    def __init__(self, env, noise_prob: float = 0.0):
        super().__init__(env)
        self.noise_prob = noise_prob
        self._rng = np.random.default_rng()

    def observation(self, obs):
        if self.noise_prob == 0.0:
            return obs
        img = obs["image"].copy()   # (H, W, 3) uint8: [type, color, state]
        H, W, _ = img.shape
        for i in range(H):
            for j in range(W):
                is_unseen = (img[i, j, 0] == 0)   # OBJECT_TO_IDX["unseen"] == 0
                if not is_unseen and self._rng.random() < self.noise_prob:
                    img[i, j, 0] = self._rng.integers(1, 11)   # random object type
                    img[i, j, 1] = self._rng.integers(0, 6)    # random color
                    img[i, j, 2] = 0
        obs["image"] = img
        return obs


class ReducedActionSpace(gym.Wrapper):
    # keep: left, right, forward, pickup, toggle
    def __init__(self, env):
        super().__init__(env)
        self.keep = [0, 1, 2, 3, 5]
        self.action_space = spaces.Discrete(len(self.keep))

    def step(self, action):
        return self.env.step(self.keep[int(action)])