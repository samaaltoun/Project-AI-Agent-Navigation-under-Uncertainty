import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.belief_map import BeliefMap, NUM_TYPES
from envs.pathfinding import astar_on_belief


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


class BeliefObsWrapper(gym.Wrapper):
    """
    Replaces MiniGrid's dict observation with a compressed belief-map tensor.

    Instead of the raw 7×7 egocentric image, the agent receives the full
    grid-sized belief map as its observation, compressed into 6 channels
    for efficient learning.

    Observation shape: (7, grid_size, grid_size)
        Channel 0 : most likely object type per cell (argmax, scaled to [0,1])
        Channel 1 : confidence (max belief probability per cell)
        Channel 2 : entropy per cell (normalised to [0,1])
        Channel 3 : agent position  (1.0 at agent cell, 0.0 elsewhere)
        Channel 4 : carrying key    (1.0 everywhere if holding key, else 0.0)
        Channel 5 : door opened     (1.0 everywhere if door is open, else 0.0)
        Channel 6 : A* planned path (1.0 on cells along the optimal route, 0.0 elsewhere)
                    A* runs on the belief map each step, planning through
                    what the agent *thinks* the grid looks like.  Under noise
                    the plan may be wrong — the NN learns when to trust it.

    This wrapper must be placed AFTER SensorNoiseWrapper and REPLACES
    ImgObsWrapper in the wrapper chain.

    Parameters
    ----------
    env : gym.Env
        Environment whose observations are MiniGrid dicts with an "image" key.
    grid_size : int
        Side length of the square grid.
    noise_prob : float
        Sensor noise probability (passed to BeliefMap for likelihood model).
    tpr : float or None
        True Positive Rate override.  If None, derived from noise_prob.
    decay : float
        Per-step belief decay toward uniform prior (0.0 = no forgetting).
    view_size : int
        Agent's egocentric field of view (default 7).
    """

    N_CHANNELS = 7   # compressed: argmax_type, confidence, entropy, agent_pos, key, door, astar_path

    def __init__(
        self,
        env: gym.Env,
        grid_size: int,
        noise_prob: float = 0.0,
        tpr: float | None = None,
        decay: float = 0.0,
        view_size: int = 7,
    ):
        super().__init__(env)
        self.grid_size = grid_size
        self._n_channels = self.N_CHANNELS  # 6

        self.bmap = BeliefMap(
            grid_size=grid_size,
            noise_prob=noise_prob,
            view_size=view_size,
            tpr=tpr,
            decay=decay,
        )

        # Override observation space: float32 tensor in channels-first (C, H, W)
        # format so we can skip VecTransposeImage (which rejects >4 channels).
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._n_channels, grid_size, grid_size),
            dtype=np.float32,
        )

    # ── gym.Wrapper overrides ──────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.bmap.reset()
        base = self.unwrapped
        self.bmap.update(obs["image"], base.agent_pos, base.agent_dir)
        return self._build_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.unwrapped
        self.bmap.update(obs["image"], base.agent_pos, base.agent_dir)
        return self._build_obs(), reward, terminated, truncated, info

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """
        Construct a compressed (6, grid_size, grid_size) observation tensor.

        Instead of passing all 11 raw belief distributions (which is too
        high-dimensional for the CNN to learn from efficiently), we compress
        the belief into three informative summary channels plus three
        binary state channels.

        Channels-first format so we can skip VecTransposeImage.
        """
        base = self.unwrapped
        gs = self.grid_size
        belief = self.bmap.belief  # (gs, gs, NUM_TYPES)

        obs = np.zeros((self._n_channels, gs, gs), dtype=np.float32)

        # Channel 0: most likely object type, scaled to [0, 1]
        #   argmax over the type axis, then divide by (NUM_TYPES-1)
        #   so the CNN sees a smooth signal: wall≈0.18, door≈0.36, goal≈0.73, etc.
        obs[0, :, :] = belief.argmax(axis=2).astype(np.float32) / max(NUM_TYPES - 1, 1)

        # Channel 1: confidence (max probability per cell)
        #   High = agent is certain about this cell; low = very uncertain
        obs[1, :, :] = belief.max(axis=2).astype(np.float32)

        # Channel 2: normalised entropy per cell  [0 = certain, 1 = max uncertain]
        entropy = self.bmap.entropy_map()  # (gs, gs), in bits
        max_entropy = np.log2(max(NUM_TYPES - 1, 1))  # ~3.32 bits
        obs[2, :, :] = np.clip(entropy / max_entropy, 0.0, 1.0).astype(np.float32)

        # Channel 3: agent position (binary)
        ax, ay = int(base.agent_pos[0]), int(base.agent_pos[1])
        if 0 <= ax < gs and 0 <= ay < gs:
            obs[3, ax, ay] = 1.0

        # Channel 4: carrying key (uniform flag)
        if base.carrying is not None and base.carrying.type == "key":
            obs[4, :, :] = 1.0

        # Channel 5: door has been opened (uniform flag)
        door = base.grid.get(*base.door_pos)
        door_is_open = door is not None and door.type == "door" and door.is_open
        if door_is_open:
            obs[5, :, :] = 1.0

        # Channel 6: A* planned path on the belief map
        #   Build a type_map from the belief (argmax at each cell),
        #   then run A* through the three sub-goals: key → door → goal.
        #   The NN sees the full planned route and learns when to follow it.
        type_map = belief.argmax(axis=2)  # (gs, gs) int array of predicted types
        agent_pos = (ax, ay)
        has_key = base.carrying is not None and base.carrying.type == "key"

        path_cells = self._plan_astar_path(
            type_map, agent_pos, has_key, door_is_open, base
        )
        for (px, py) in path_cells:
            if 0 <= px < gs and 0 <= py < gs:
                obs[6, px, py] = 1.0

        return obs

    def _plan_astar_path(
        self,
        type_map,
        agent_pos: tuple,
        has_key: bool,
        door_is_open: bool,
        base,
    ) -> list[tuple[int, int]]:
        """
        Run A* on the belief type map to plan the full key→door→goal route.

        Adapts to the agent's current progress:
        - If it hasn't picked up the key yet: plan start → key → door → goal
        - If it has the key but door is closed: plan start → door → goal
        - If the door is open: plan start → goal

        Uses the *believed* positions of key/door/goal where possible.
        Falls back to the real positions from the environment (which the
        agent always knows — it remembers where it saw them).
        """
        import numpy as np

        gs = self.grid_size
        KEY_TYPE, DOOR_TYPE, GOAL_TYPE = 5, 4, 8

        # Find believed positions of key, door, goal by scanning the type map.
        # If the belief map doesn't clearly show them (e.g. key already picked up),
        # fall back to the real position from the environment.
        def _find_believed_pos(obj_type: int, fallback: tuple) -> tuple:
            matches = np.argwhere(type_map == obj_type)
            if len(matches) == 0:
                return fallback
            # Pick the match closest to the agent (most likely the real one)
            dists = np.abs(matches[:, 0] - agent_pos[0]) + np.abs(matches[:, 1] - agent_pos[1])
            best = matches[dists.argmin()]
            return (int(best[0]), int(best[1]))

        key_pos  = _find_believed_pos(KEY_TYPE, tuple(base.key_pos))
        door_pos = _find_believed_pos(DOOR_TYPE, tuple(base.door_pos))
        goal_pos = _find_believed_pos(GOAL_TYPE, tuple(base.goal_pos))

        full_path: list[tuple[int, int]] = []

        if not has_key:
            # Segment 1: agent → key  (door is locked = blocked)
            seg = astar_on_belief(type_map, agent_pos, key_pos,
                                  blocked_positions={door_pos})
            if seg:
                full_path.extend(seg)
            # Segment 2: key → door
            seg = astar_on_belief(type_map, key_pos, door_pos)
            if seg:
                full_path.extend(seg[1:])  # skip duplicate key_pos
            # Segment 3: door → goal
            seg = astar_on_belief(type_map, door_pos, goal_pos)
            if seg:
                full_path.extend(seg[1:])  # skip duplicate door_pos

        elif not door_is_open:
            # Segment 1: agent → door
            seg = astar_on_belief(type_map, agent_pos, door_pos)
            if seg:
                full_path.extend(seg)
            # Segment 2: door → goal
            seg = astar_on_belief(type_map, door_pos, goal_pos)
            if seg:
                full_path.extend(seg[1:])

        else:
            # Door is open — just go to goal
            seg = astar_on_belief(type_map, agent_pos, goal_pos)
            if seg:
                full_path.extend(seg)

        return full_path
