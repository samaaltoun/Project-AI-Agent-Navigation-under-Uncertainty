from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


@dataclass
class ShapingConfig:
    reward_pickup_key: float = 0.2
    reward_open_door: float = 0.3
    reward_goal: float = 1.0
    step_penalty: float = -0.0001  # small penalty to encourage shorter solutions


class SimpleDoorEnv(MiniGridEnv):
    """
    Simple map:
    - Vertical wall at x=5
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


        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size ** 2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal."

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Goal
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)
        self.put_obj(Wall(), 3, 0)
        self.put_obj(Wall(), 3, 1)
        #self.put_obj(Wall(), 3, 2)
        self.put_obj(Wall(), 3, 4)
        self.put_obj(Wall(), 3, 3)
        self.put_obj(Wall(), 3, 5)
        self.put_obj(Wall(), 3, 6)
        self.put_obj(Wall(), 3, 7)
        self.put_obj(Wall(), 3, 8)


        # Agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = self._gen_mission()

    def step(self, action: int):
        """
        Adds reward shaping on top of MiniGrid's reward.
        MiniGrid reward usually gives >0 when reaching goal.
        """
        (a_x0, a_y0) = self.agent_pos
        obs, reward, terminated, truncated, info = super().step(action)

        def manhattan_distance(x, y):
            return abs(x[0] - y[0]) + abs(x[1] - y[1])

        shaped = 0.0
        # step penalty (small)
        shaped += self.shaping.step_penalty

        if a_x0 > 2:
            # Reward based on getting closer to the goal
            target = self.goal_pos
            dist0 = manhattan_distance((a_x0, a_y0), target)
            dist1 = manhattan_distance(self.agent_pos, target)
            shaped += 0.1 * (dist0 - dist1)
        else:
            shaped -= 0.001

        # event: reached goal
        if terminated:
            # MiniGrid already gives reward; we add a consistent “goal reward” if you want:
            shaped += self.shaping.reward_goal

        total_reward = float(reward + shaped)
        info = dict(info)
        info["shaped_reward"] = shaped
        info["is_success"] = bool(terminated)

        return obs, total_reward, terminated, truncated, info


# Optional: register so gym.make() works
def register_env():
    gym.register(
        id="SimpleGoalEnv-v0",
        entry_point="envs.simple_goal_env:SimpleDoorEnv",
    )