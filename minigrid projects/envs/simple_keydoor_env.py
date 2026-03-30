from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv


@dataclass
class ShapingConfig:
    reward_pickup_key: float = 2.0
    reward_open_door: float  = 1.5
    reward_goal: float       = 5.0
    step_penalty: float      = -0.001


class SimpleKeyDoorEnv(MiniGridEnv):
    """
    Single-room layout with lava hazards:
    - One vertical wall dividing left / right sections
    - Key on the left, locked door in the wall, goal on the right
    - Agent starts on the left
    - Lava tiles scattered across the grid
    """

    def __init__(
        self,
        size: int = 13,
        agent_start_pos: Tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        shaping: ShapingConfig | None = None,
        n_lava: int = 0,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.shaping = shaping or ShapingConfig()
        self.n_lava = n_lava

        self._picked_key  = False
        self._opened_door = False

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
        return "Pick up the key, open the door, reach the goal."

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._picked_key  = False
        self._opened_door = False
        return obs, info

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # ── One vertical wall ─────────────────────────────────────────────────
        wall_x = self._rand_int(3, width - 3)
        for y in range(height):
            self.grid.set(wall_x, y, Wall())

        # ── Door ──────────────────────────────────────────────────────────────
        door_y = self._rand_int(1, height - 1)
        self.door_pos = (wall_x, door_y)
        self.grid.set(wall_x, door_y, Door(COLOR_NAMES[0], is_locked=True))

        # ── Key on the left ───────────────────────────────────────────────────
        self.key_pos = self.place_obj(
            Key(COLOR_NAMES[0]),
            top=(1, 1),
            size=(wall_x - 1, height - 2),
        )

        # ── Goal on the right ─────────────────────────────────────────────────
        self.goal_pos = self.place_obj(
            Goal(),
            top=(wall_x + 1, 1),
            size=(width - wall_x - 2, height - 2),
        )

        # ── Agent on the left ─────────────────────────────────────────────────
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent(top=(1, 1), size=(wall_x - 1, height - 2))

        # ── Lava scattered across the whole interior ──────────────────────────
        for _ in range(self.n_lava):
            self.place_obj(Lava(), top=(1, 1), size=(width - 2, height - 2))

        # ── Guarantee critical objects are always safely approachable ─────────
        # Clear lava from the 4 orthogonal neighbours of the key and goal,
        # and from the cells directly flanking the door.
        critical_cells: set = set()
        for ox, oy in (self.key_pos, self.goal_pos):
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                critical_cells.add((ox + dx, oy + dy))
        critical_cells.add((wall_x - 1, door_y))
        critical_cells.add((wall_x + 1, door_y))

        for cx, cy in critical_cells:
            if 0 < cx < width - 1 and 0 < cy < height - 1:
                cell = self.grid.get(cx, cy)
                if cell is not None and cell.type == "lava":
                    self.grid.set(cx, cy, None)

        self.mission = self._gen_mission()

    def step(self, action: int):
        def manhattan_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_target():
            if not self._picked_key:
                return self.key_pos
            if not self._opened_door:
                return self.door_pos
            return self.goal_pos

        gamma_shaping = 0.99
        k = 0.05

        # ── PRE-STEP ──────────────────────────────────────────────────────────
        pos0            = tuple(self.agent_pos)
        phi0            = -manhattan_distance(pos0, get_target())
        carrying_before = self.carrying

        fwd0       = tuple(self.front_pos)
        obj_ahead0 = self.grid.get(fwd0[0], fwd0[1])

        door = self.grid.get(*self.door_pos)
        was_open = door is not None and door.type == "door" and door.is_open

        ahead_is_locked_door = (
            obj_ahead0 is not None
            and obj_ahead0.type == "door"
            and obj_ahead0.is_locked
        )
        has_key = self.carrying is not None and self.carrying.type == "key"

        shaped = float(self.shaping.step_penalty)

        # ── ENV TRANSITION ────────────────────────────────────────────────────
        obs, reward, terminated, truncated, info = super().step(action)

        # ── POST-STEP ─────────────────────────────────────────────────────────
        pos1 = tuple(self.agent_pos)

        if action == self.actions.forward and pos1 == pos0:
            shaped -= 0.05

        door = self.grid.get(*self.door_pos)
        is_open = door is not None and door.type == "door" and door.is_open
        opened_this_step = (not was_open) and is_open

        # ── EVENT REWARDS ─────────────────────────────────────────────────────
        if (not self._picked_key) and self.carrying is not None and self.carrying.type == "key":
            self._picked_key = True
            shaped += self.shaping.reward_pickup_key

        if (not self._opened_door) and is_open:
            self._opened_door = True
            shaped += self.shaping.reward_open_door

        if terminated and tuple(self.agent_pos) == tuple(self.goal_pos):
            shaped += self.shaping.reward_goal

        # ── POTENTIAL-BASED SHAPING ───────────────────────────────────────────
        phi1    = -manhattan_distance(pos1, get_target())
        shaped += k * (gamma_shaping * phi1 - phi0)

        # ── DOOR BOTTLENECK SHAPING ───────────────────────────────────────────
        if ahead_is_locked_door and has_key:
            if action == self.actions.toggle:
                shaped += 0.3 if opened_this_step else -0.05
            elif action == self.actions.forward and pos1 == pos0:
                shaped -= 0.05

        if action == self.actions.toggle and not opened_this_step and not (ahead_is_locked_door and has_key):
            shaped -= 0.02

        if action == self.actions.pickup and self.carrying == carrying_before:
            shaped -= 0.05

        if pos1 == pos0 and action in (self.actions.left, self.actions.right, self.actions.toggle):
            shaped -= 0.01

        total_reward = float(reward + shaped)

        info = dict(info)
        info["shaped_reward"] = shaped
        info["picked_key"]    = self._picked_key
        info["opened_door"]   = self._opened_door
        info["agent_pos"]     = tuple(self.agent_pos)
        info["goal_pos"]      = tuple(self.goal_pos)
        info["is_success"]    = (tuple(self.agent_pos) == tuple(self.goal_pos))

        return obs, total_reward, terminated, truncated, info


def register_env():
    if "SimpleKeyDoor-v0" not in gym.envs.registry:
        gym.register(
            id="SimpleKeyDoor-v0",
            entry_point="envs.simple_keydoor_env:SimpleKeyDoorEnv",
        )
