# envs/variational_goal_env.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


@dataclass
class ShapingConfig:
    reward_pickup_key: float = 2.0
    reward_open_door: float = 1.5
    reward_goal: float = 5.0
    step_penalty: float = -0.001


def sample_unique(region: list, forbidden: set) -> tuple:
    """Sample a unique position from a region, avoiding forbidden positions."""
    candidates = [pos for pos in region if pos not in forbidden]
    if not candidates:
        raise ValueError("No valid positions available in region.")
    idx = np.random.randint(len(candidates))
    return candidates[idx]


class VariationalGoalEnv(MiniGridEnv):
    """
    Variational goal environment with randomised layout each episode:
    - Random vertical wall splitting the grid in two regions
    - Locked door placed randomly on the wall
    - Key placed randomly in the left region
    - Goal placed randomly in the right region
    - Agent placed in a random empty cell (left region)
    """

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int | None = None,
        shaping: ShapingConfig | None = None,
        **kwargs,
    ):
        self.shaping = shaping or ShapingConfig()

        # Track shaping events per episode
        self._picked_key = False
        self._opened_door = False
        self.door_pos = None
        self.key_pos = None
        self.goal_pos = None

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * grid_size ** 2

        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Find the key, open the door, and reach the goal."

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._picked_key = False
        self._opened_door = False
        return obs, info

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- 1. Vertical wall splitting the grid ---
        split_col = width // 2
        for y in range(1, height - 1):
            self.grid.set(split_col, y, Wall())

        # --- 2. Door randomly placed on the wall ---
        door_y = int(np.random.randint(1, height - 1))
        self.door_pos = (split_col, door_y)
        self.grid.set(split_col, door_y, Door("yellow", is_locked=True))

        # --- 3. Define two regions ---
        region_left = [(x, y) for x in range(1, split_col) for y in range(1, height - 1)]
        region_right = [(x, y) for x in range(split_col + 1, width - 1) for y in range(1, height - 1)]

        forbidden: set = {self.door_pos}

        # --- 4. Key in left region (using sample_unique) ---
        self.key_pos = sample_unique(region_left, forbidden)
        forbidden.add(self.key_pos)
        self.grid.set(self.key_pos[0], self.key_pos[1], Key("yellow"))

        # --- 5. Goal in right region (using sample_unique) ---
        goal_pos = sample_unique(region_right, forbidden)
        forbidden.add(goal_pos)
        self.goal_pos = goal_pos
        self.grid.set(goal_pos[0], goal_pos[1], Goal())

        # --- 6. Agent in left region (using sample_unique) ---
        agent_pos = sample_unique(region_left, forbidden)
        self.agent_pos = np.array(agent_pos)
        self.agent_dir = int(np.random.randint(0, 4))

        self.mission = self._gen_mission()

    def step(self, action: int):
        """
        Reward shaping mirroring SimpleKeyDoorEnv:
        - Potential-based shaping toward the current sub-goal (key -> door -> goal)
        - Event rewards: pick up key, open door, reach goal
        - Door bottleneck: reward toggle only if it actually opens the door
        - Penalise bumping / pointless toggles
        """

        def manhattan_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_target(picked_key: bool, opened_door: bool):
            if not picked_key:
                return self.key_pos
            if not opened_door:
                return self.door_pos
            return self.goal_pos

        gamma_shaping = 0.99
        k = 0.05

        # ----- PRE-STEP STATE -----
        pos0 = tuple(self.agent_pos)
        target0 = get_target(self._picked_key, self._opened_door)
        phi0 = -manhattan_distance(pos0, target0)

        fwd0 = tuple(self.front_pos)
        obj_ahead0 = self.grid.get(fwd0[0], fwd0[1])

        door0 = self.grid.get(*self.door_pos)
        was_open0 = (door0 is not None and door0.type == "door" and door0.is_open)

        ahead_is_locked_door0 = (
            obj_ahead0 is not None
            and obj_ahead0.type == "door"
            and obj_ahead0.is_locked
        )
        has_key0 = (self.carrying is not None and self.carrying.type == "key")

        shaped = float(self.shaping.step_penalty)

        # ----- ENV TRANSITION -----
        obs, reward, terminated, truncated, info = super().step(action)

        # ----- POST-STEP STATE -----
        pos1 = tuple(self.agent_pos)

        # Penalise bumping forward
        if action == self.actions.forward and pos1 == pos0:
            shaped -= 0.05

        door1 = self.grid.get(*self.door_pos)
        is_open1 = (door1 is not None and door1.type == "door" and door1.is_open)
        opened_this_step = (not was_open0) and is_open1

        # ----- EVENT REWARDS -----
        if (not self._picked_key) and (self.carrying is not None) and (self.carrying.type == "key"):
            self._picked_key = True
            shaped += self.shaping.reward_pickup_key

        if (not self._opened_door) and is_open1:
            self._opened_door = True
            shaped += self.shaping.reward_open_door

        if terminated:
            shaped += self.shaping.reward_goal

        # ----- POTENTIAL-BASED SHAPING -----
        target1 = get_target(self._picked_key, self._opened_door)
        phi1 = -manhattan_distance(pos1, target1)
        shaped += k * (gamma_shaping * phi1 - phi0)

        # ----- DOOR BOTTLENECK SHAPING -----
        if ahead_is_locked_door0 and has_key0:
            if action == self.actions.toggle:
                if opened_this_step:
                    shaped += 0.3
                else:
                    shaped -= 0.05
            elif action == self.actions.forward and pos1 == pos0:
                shaped -= 0.05

        # Penalise pointless toggles anywhere
        if action == self.actions.toggle and not opened_this_step and not (ahead_is_locked_door0 and has_key0):
            shaped -= 0.02

        # Penalise spinning/toggling in place
        if pos1 == pos0 and action in (self.actions.left, self.actions.right, self.actions.toggle):
            shaped -= 0.01

        total_reward = float(reward + shaped)

        info = dict(info)
        info["shaped_reward"] = shaped
        info["picked_key"] = self._picked_key
        info["opened_door"] = self._opened_door
        info["agent_pos"] = tuple(self.agent_pos)
        info["goal_pos"] = tuple(self.goal_pos)
        info["is_success"] = bool(terminated and tuple(self.agent_pos) == tuple(self.goal_pos))

        return obs, total_reward, terminated, truncated, info


def register_env():
    env_id = "VariationalGoal-v0"
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point="envs.variational_goal_env:VariationalGoalEnv",
        )
