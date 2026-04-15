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


class BeliefMapObsWrapper(gym.Wrapper):
    """
    Replaces the raw MiniGrid image observation with the agent's belief map
    plus explicit agent-position, goal-position, and direction channels.

    Output shape: (9, H, W) float32 — already channel-first,
    so VecTransposeImage is NOT needed. Channels are:
        0 : wall probability        (from belief map)
        1 : key  probability        (from belief map)
        2 : door probability        (from belief map)
        3 : goal probability        (from belief map)
        4 : agent position          (1.0 at agent cell, 0.0 elsewhere)
        5 : goal position           (1.0 at goal  cell, 0.0 elsewhere)
        6 : agent direction         (constant map filled with dir/3.0,
                                     so 0=right, 0.33=down, 0.67=left, 1.0=up)
        7 : carrying key flag       (constant map: 1.0 if holding key, 0.0 otherwise)
        8 : door open flag          (constant map: 1.0 if door is open, 0.0 otherwise)

    Channels 7 and 8 give the agent explicit task-progress state that it
    cannot reliably infer from the belief map under heavy observation noise.
    """

    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped
        size = base.size
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9, size, size),
            dtype=np.float32,
        )

    def _build_obs(self) -> np.ndarray:
        base = self.env.unwrapped
        size = base.size

        # Channels 0-3: belief map
        belief = base.belief.stack()                          # (4, H, W)

        # Channel 4: agent position indicator
        agent_ch = np.zeros((size, size), dtype=np.float32)
        ax, ay = base.agent_pos
        agent_ch[int(ay), int(ax)] = 1.0

        # Channel 5: goal position indicator
        goal_ch = np.zeros((size, size), dtype=np.float32)
        gx, gy = base.goal_pos
        goal_ch[int(gy), int(gx)] = 1.0

        # Channel 6: agent direction (scalar broadcast to full map)
        # MiniGrid directions: 0=right, 1=down, 2=left, 3=up → normalise to [0, 1]
        dir_ch = np.full((size, size), base.agent_dir / 3.0, dtype=np.float32)

        # Channel 7: carrying key flag (1.0 if agent is holding the key)
        carrying = 1.0 if (base.carrying is not None and base.carrying.type == "key") else 0.0
        carrying_ch = np.full((size, size), carrying, dtype=np.float32)

        # Channel 8: door open flag (1.0 if door is currently open)
        door_obj = base.grid.get(*base.door_pos) if hasattr(base, "door_pos") else None
        door_open = 1.0 if (door_obj is not None and door_obj.type == "door" and door_obj.is_open) else 0.0
        door_ch = np.full((size, size), door_open, dtype=np.float32)

        return np.concatenate(
            [belief, agent_ch[np.newaxis], goal_ch[np.newaxis],
             dir_ch[np.newaxis], carrying_ch[np.newaxis], door_ch[np.newaxis]], axis=0
        )  # (9, H, W)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._build_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._build_obs(), reward, terminated, truncated, info


class ReducedActionSpace(gym.Wrapper):
    # keep: left, right, forward, pickup, toggle
    def __init__(self, env):
        super().__init__(env)
        self.keep = [0, 1, 2, 3, 5]
        self.action_space = spaces.Discrete(len(self.keep))

    def step(self, action):
        return self.env.step(self.keep[int(action)])