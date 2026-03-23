from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from minigrid.core import grid

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


@dataclass
class ShapingConfig:
    reward_pickup_key: float = 2.0
    reward_open_door: float = 1.5
    reward_goal: float = 5.0
    step_penalty: float = -0.001  # small penalty to encourage shorter solutions
 

class SimpleKeyDoorEnv(MiniGridEnv):
    """
    Simple map:
    - Vertical wall at x=5
    - Door at (5,6)
    - Key at (3,6)
    - Goal at bottom-right
    """

    def __init__(
        self,
        size: int = 10,
        agent_start_pos: Tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        shaping: ShapingConfig | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.shaping = shaping or ShapingConfig()

        # Track shaping events per episode
        self._picked_key = False
        self._opened_door = False

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Pick up the key, open the door, reach the goal."

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._picked_key = False
        self._opened_door = False
        return obs, info

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Vertical wall at x=5
        for y in range(height):
            self.grid.set(5, y, Wall())

        # Door replaces wall at (5,6)
        self.door_pos = (5, 6)
        self.key_pos = (3, 6)

        self.grid.set(self.door_pos[0], self.door_pos[1], Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(self.key_pos[0], self.key_pos[1], Key(COLOR_NAMES[0]))

        # Goal
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)

        # Agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = self._gen_mission()

    def step(self, action: int):
        """
        Reward shaping for SimpleKeyDoor:
        - Potential-based shaping toward the current sub-goal (key -> door -> goal)
        - Event rewards: pick up key, open door, reach goal
        - Penalize bumping into obstacles (forward but no movement)
        - Door bottleneck shaping:
            * Reward toggle ONLY if it actually opens the locked door this step
            * Penalize toggle otherwise (prevents toggle-spam)
            * Penalize forward into locked door when carrying key (prevents forward-spam at door)
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

        # ----- PRE-STEP STATE (s0) -----
        pos0 = tuple(self.agent_pos)
        target0 = get_target(self._picked_key, self._opened_door)
        phi0 = -manhattan_distance(pos0, target0)

        # what's directly ahead BEFORE acting
        fwd0 = tuple(self.front_pos)
        obj_ahead0 = self.grid.get(fwd0[0], fwd0[1])

        # door state BEFORE acting
        door0 = self.grid.get(*self.door_pos)
        was_open0 = (door0 is not None and door0.type == "door" and door0.is_open)

        # bottleneck condition BEFORE acting
        ahead_is_locked_door0 = (
                obj_ahead0 is not None
                and obj_ahead0.type == "door"
                and obj_ahead0.is_locked
        )
        has_key0 = (self.carrying is not None and self.carrying.type == "key")

        # base shaped reward
        shaped = float(self.shaping.step_penalty)

        # ----- ENV TRANSITION -----
        obs, reward, terminated, truncated, info = super().step(action)

        # ----- POST-STEP STATE (s1) -----
        pos1 = tuple(self.agent_pos)

        # Penalize bumping forward (prevents "stuck pushing")
        if action == self.actions.forward and pos1 == pos0:
            shaped -= 0.05

        # Door state AFTER acting
        door1 = self.grid.get(*self.door_pos)
        is_open1 = (door1 is not None and door1.type == "door" and door1.is_open)
        opened_this_step = (not was_open0) and is_open1

        # ----- EVENT REWARDS -----
        # picked up key (ONLY if actually carrying key)
        if (not self._picked_key) and (self.carrying is not None) and (self.carrying.type == "key"):
            self._picked_key = True
            shaped += self.shaping.reward_pickup_key

        # opened door event (door must be open)
        if (not self._opened_door) and is_open1:
            self._opened_door = True
            shaped += self.shaping.reward_open_door

        # reached goal
        if terminated:
            shaped += self.shaping.reward_goal

        # ----- POTENTIAL-BASED SHAPING (correct form) -----
        target1 = get_target(self._picked_key, self._opened_door)
        phi1 = -manhattan_distance(pos1, target1)
        shaped += k * (gamma_shaping * phi1 - phi0)

        # ----- DOOR BOTTLENECK SHAPING (prevents forward/toggle spam) -----
        # If locked door ahead and we have key:
        #   - reward toggle ONLY if it actually opened the door now
        #   - penalize forward into the locked door
        if ahead_is_locked_door0 and has_key0:
            if action == self.actions.toggle:
                if opened_this_step:
                    shaped += 0.3
                else:
                    shaped -= 0.05
            elif action == self.actions.forward and pos1 == pos0:
                shaped -= 0.05

        # Penalize pointless toggles in general (stops "toggle forever" anywhere)
        if action == self.actions.toggle and not opened_this_step and not (ahead_is_locked_door0 and has_key0):
            shaped -= 0.02

        # If position didn't change, discourage wasting steps by spinning/toggling
        if pos1 == pos0 and action in (self.actions.left, self.actions.right, self.actions.toggle):
            shaped -= 0.01

        total_reward = float(reward + shaped)

        info = dict(info)
        info["shaped_reward"] = shaped
        info["picked_key"] = self._picked_key
        info["opened_door"] = self._opened_door
        info["agent_pos"] = tuple(self.agent_pos)
        info["goal_pos"] = tuple(self.goal_pos)
        info["is_success"] = (tuple(self.agent_pos) == tuple(self.goal_pos))

        return obs, total_reward, terminated, truncated, info

# Optional: register so gym.make() works
def register_env():
    gym.register(
        id="SimpleKeyDoor-v0",
        entry_point="envs.simple_keydoor_env:SimpleKeyDoorEnv",
    )