from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import gymnasium as gym
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

#config dataclasses

@dataclass
class SensorConfig:
    accuracy: float = 1.0       # perfect within-window accuracy
    noise_prob: float = 0.0     # prob of dropout per cell per step — applied inside belief update
    tpr: float = 0.85           # prob that sensor detects something when it actually exists
    tnr: float = 0.90           # prob sensor correctly classifies "not present"
    wall_prior: float = 0.5
    key_prior: float = 0.05
    door_prior: float = 0.05
    goal_prior: float = 0.05
    decay: float = 0.001        # slower forgetting — needed under heavy observation dropout


@dataclass
class RewardConfig:
    reward_pickup_key: float = 8.0
    reward_open_door: float = 5.0
    reward_goal: float = 15.0
    step_penalty: float = -0.001
    bump_penalty: float = -0.05
    pointless_toggle_penalty: float = -0.5   # heavy penalty toggling uselessly must not be viable
    reward_progress: float = 1.0             # reward per step closer to current sub-goal


#belief map 

class BeliefMap:
    #Stores one Bernoulli probability map per object type

    def __init__(self, width: int, height: int, cfg: SensorConfig):
        self.width = width
        self.height = height
        self.cfg = cfg
        self.wall = np.full((height, width), cfg.wall_prior, dtype=np.float32)
        self.key = np.full((height, width), cfg.key_prior, dtype=np.float32)
        self.door = np.full((height, width), cfg.door_prior, dtype=np.float32)
        self.goal = np.full((height, width), cfg.goal_prior, dtype=np.float32)
        self.maps = {
            "wall": self.wall,
            "key": self.key,
            "door": self.door,
            "goal": self.goal,
        }

    def bayes_update(self, prior: float, seen: int) -> float:
        prior = max(1e-6, min(1.0 - 1e-6, float(prior)))
        tpr = self.cfg.tpr
        tnr = self.cfg.tnr

        if seen == 1:
            num = tpr * prior
            den = tpr * prior + (1.0 - tnr) * (1.0 - prior)
        else:
            num = (1.0 - tpr) * prior #we can change tpr over time 
            den = (1.0 - tpr) * prior + tnr * (1.0 - prior)

        return float(num / (den + 1e-8))

    def renormalize_cell(self, x: int, y: int) -> None:
        total = (
            self.wall[y, x]
            + self.key[y, x]
            + self.door[y, x]
            + self.goal[y, x]
        )
        if total > 1.0:
            for grid in self.maps.values():
                grid[y, x] /= total

    def update_cell(self, x: int, y: int, label: str) -> None:
        for name, grid in self.maps.items():
            if label == name:
                seen = 1
            else:
                seen = 0
            grid[y, x] = self.bayes_update(grid[y, x], seen)
            if name != label:
                grid[y, x] *= 0.1

        self.renormalize_cell(x, y)

    def apply_decay(self, observed_mask: np.ndarray) -> None: #add various levels, more forgetting 
        alpha = self.cfg.decay
        if alpha <= 0:
            return
        for grid in self.maps.values():
            grid[~observed_mask] *= (1.0 - alpha)

    def set_border_walls(self) -> None:
        border_values = {"wall": 1.0, "key": 0.0, "door": 0.0, "goal": 0.0}
        for name, grid in self.maps.items():
            grid[0, :]  = border_values[name]
            grid[-1, :] = border_values[name]
            grid[:, 0]  = border_values[name]
            grid[:, -1] = border_values[name]

    def stack(self) -> np.ndarray:
        return np.stack(
            [self.wall, self.key, self.door, self.goal], axis=0
        ).astype(np.float32)


#environment 

class NoisyBeliefKeyEnv(MiniGridEnv):
    
    #Partial obs environment with:
    #noisy sensor labels
    #multi-Bernoulli belief map
    #key, door, and goal objects
    def __init__(
        self,
        size: int = 8,
        agent_start_pos: Tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        agent_view_size: int = 5,
        num_extra_walls: int = 0,
        randomize_layout: bool = True,
        max_steps: int | None = None,
        sensor_cfg: SensorConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        **kwargs,):
        
        self.size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.num_extra_walls = num_extra_walls
        self.randomize_layout = randomize_layout
        if sensor_cfg is None:
            sensor_cfg = SensorConfig()
        if reward_cfg is None:
            reward_cfg = RewardConfig()
        self.sensor_cfg = sensor_cfg
        self.reward_cfg = reward_cfg

        self.split_x = size // 2
        #default positions (used when randomize layout is set to False, done for test trials)
        self.door_pos = (self.split_x, size - 4)
        self.key_pos = (max(2, self.split_x - 2), size - 4)
        self.goal_pos = (size - 2, size - 2)
        self.picked_key = False
        self.opened_door = False
        self.belief: BeliefMap | None = None
        self.last_sensor_report = []
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 8 * size * size   # doubled — agent needs more time under noise

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=agent_view_size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "Use noisy observations to find the key, open the door, and reach the goal."

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Vertical barrier wall
        for y in range(1, height - 1):
            self.grid.set(self.split_x, y, Wall())

        if self.randomize_layout:
            # Random agent start — left side only (barrier is active)
            start_cells = [
                (x, y)
                for x in range(1, self.split_x)
                for y in range(1, height - 1)
            ]
            si = int(self.np_random.integers(0, len(start_cells)))
            self.agent_start_pos = start_cells[si]

            # Random door along barrier
            door_y = int(self.np_random.integers(2, height - 2))
            self.door_pos = (self.split_x, door_y)

            # Random key on left side
            left_cells = [
                (x, y)
                for x in range(1, self.split_x)
                for y in range(1, height - 1)
                if (x, y) != tuple(self.agent_start_pos)
            ]
            ki = int(self.np_random.integers(0, len(left_cells)))
            self.key_pos = left_cells[ki]

            # Random goal — right side only (barrier is active)
            goal_cells = [
                (x, y)
                for x in range(self.split_x + 1, width - 1)
                for y in range(1, height - 1)
            ]
            gi = int(self.np_random.integers(0, len(goal_cells)))
            self.goal_pos = goal_cells[gi]

        #place the door and key
        self.grid.set(self.door_pos[0], self.door_pos[1], Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(self.key_pos[0],  self.key_pos[1],  Key(COLOR_NAMES[0]))

        self.put_obj(Goal(), *self.goal_pos)

        #place agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        #extra random walls (with connectivity validation)
        if self.num_extra_walls > 0:
            reserved = set()
            for x in range(width):
                reserved.add((x, 0))
                reserved.add((x, height - 1))
            for y in range(height):
                reserved.add((0, y))
                reserved.add((width - 1, y))
            #reserve the  barrier and door/key cells
            for y in range(1, height - 1):
                reserved.add((self.split_x, y))
            reserved.add(self.door_pos)
            reserved.add(self.key_pos)
            reserved.add(self.goal_pos)
            reserved.add(tuple(self.agent_pos))

            free = [
                (x, y)
                for x in range(1, width - 1)
                for y in range(1, height - 1)
                if (x, y) not in reserved
            ]
            n_place = min(self.num_extra_walls, len(free))

            #Retry up to 10 times until a connected layout is found
            placed: List[Tuple[int, int]] = []
            for _ in range(10):
                indices = self.np_random.choice(len(free), size=n_place, replace=False)
                placed = [free[int(i)] for i in indices]
                for wx, wy in placed:
                    self.grid.set(wx, wy, Wall())
                if self._check_connectivity(width, height):
                    break
                #remove and retry if not connected 
                for wx, wy in placed:
                    self.grid.set(wx, wy, None)
                placed = []

        self.mission = self._gen_mission()

    #checking for connectivity 

    def _check_connectivity(self, width: int, height: int) -> bool:
        """
        BFS from agent start to verify that key, door, and goal are all reachable.
        Doors are treated as passable (the agent can open them with the key).
        Returns True if the layout is fully connected.
        """
        start = tuple(self.agent_start_pos)
        targets: Set[Tuple[int, int]] = {
            tuple(self.key_pos),
            tuple(self.door_pos),
            tuple(self.goal_pos),
        }
        visited: Set[Tuple[int, int]] = {start}
        queue: deque = deque([start])
        found: Set[Tuple[int, int]] = set()

        while queue:
            x, y = queue.popleft()
            if (x, y) in targets:
                found.add((x, y))
            if found == targets:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                npos = (nx, ny)
                if npos in visited:
                    continue
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                cell = self.grid.get(nx, ny)
                #able to pass through:  empty cell, key, goal, or door treated as openable
                if cell is None or cell.type in ("key", "goal", "door"):
                    visited.add(npos)
                    queue.append(npos)

        return found == targets

   #sensors 

    def _true_label(self, x: int, y: int) -> str:
        obj = self.grid.get(x, y)
        if obj is None:
            return "empty"
        if obj.type == "wall":
            return "wall"
        if obj.type == "key":
            return "key"
        if obj.type == "door":
            return "door"
        if obj.type == "goal":
            return "goal"
        return "empty"

    def _noisy_label(self, true_label: str) -> str:
        labels = ["empty", "wall", "key", "door", "goal"]
        # Step 1: observation dropout — corrupts cell before belief update
        if self.sensor_cfg.noise_prob > 0.0:
            if self.np_random.random() < self.sensor_cfg.noise_prob:
                wrong_labels = [lab for lab in labels if lab != true_label]
                idx = int(self.np_random.integers(0, len(wrong_labels)))
                return wrong_labels[idx]
        # Step 2: sensor accuracy noise
        if self.np_random.random() < self.sensor_cfg.accuracy:
            return true_label
        wrong_labels = [lab for lab in labels if lab != true_label]
        idx = int(self.np_random.integers(0, len(wrong_labels)))
        return wrong_labels[idx]

    def _get_visible_cells(self):
        visible = []
        ax, ay = self.agent_pos
        radius = self.agent_view_size // 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = ax + dx
                y = ay + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    visible.append((x, y))
        return visible

    def _update_belief(self) -> None:
        assert self.belief is not None

        observed_mask = np.zeros((self.height, self.width), dtype=bool)
        sensor_report = []

        for x, y in self._get_visible_cells():
            observed_mask[y, x] = True

            if (x, y) == tuple(self.agent_pos):
                true_label = "empty"
                noisy_label = "empty"
            else:
                true_label = self._true_label(x, y)
                noisy_label = self._noisy_label(true_label)

            self.belief.update_cell(x, y, noisy_label)
            sensor_report.append(
                {
                    "pos": (x, y),
                    "true_label": true_label,
                    "noisy_label": noisy_label,
                }
            )

        self.belief.apply_decay(observed_mask)
        self.belief.set_border_walls()
        self.last_sensor_report = sensor_report

    def _build_info(self) -> Dict[str, Any]:
        assert self.belief is not None

        door = self.grid.get(*self.door_pos)
        door_open = bool(door is not None and door.type == "door" and door.is_open)
        carrying_key = bool(self.carrying is not None and self.carrying.type == "key")

        return {
            "belief": self.belief.stack().copy(),
            "wall_prob": self.belief.wall.copy(),
            "key_prob": self.belief.key.copy(),
            "door_prob": self.belief.door.copy(),
            "goal_prob": self.belief.goal.copy(),
            "sensor_report": list(self.last_sensor_report),
            "agent_pos": tuple(int(v) for v in self.agent_pos),
            "agent_dir": int(self.agent_dir),
            "picked_key": self.picked_key,
            "opened_door": self.opened_door,
            "carrying_key": carrying_key,
            "door_open": door_open,
            "goal_pos": tuple(int(v) for v in self.goal_pos),
            "is_success": tuple(self.agent_pos) == tuple(self.goal_pos),
        }

    #gym api 

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)

        self.picked_key = False
        self.opened_door = False

        self.belief = BeliefMap(self.width, self.height, self.sensor_cfg)
        self.belief.set_border_walls()
        self._update_belief()

        info = dict(info)
        info.update(self._build_info())
        return obs, info

    def step(self, action: int):
        pos0 = tuple(self.agent_pos)

        door0 = self.grid.get(*self.door_pos)
        was_open0 = bool(door0 is not None and door0.type == "door" and door0.is_open)

        obs, reward, terminated, truncated, info = super().step(action)

        shaped = float(self.reward_cfg.step_penalty)

        pos1 = tuple(self.agent_pos)
        if action == self.actions.forward and pos1 == pos0:
            shaped += self.reward_cfg.bump_penalty

        # Progress reward — full sub-goal chain: key → door → goal
        if not self.picked_key:
            target = self.key_pos
        elif not self.opened_door:
            target = self.door_pos
        else:
            target = self.goal_pos

        dist0 = abs(pos0[0] - target[0]) + abs(pos0[1] - target[1])
        dist1 = abs(pos1[0] - target[0]) + abs(pos1[1] - target[1])
        shaped += self.reward_cfg.reward_progress * (dist0 - dist1)

        # Key + door rewards
        door1 = self.grid.get(*self.door_pos)
        is_open1 = bool(door1 is not None and door1.type == "door" and door1.is_open)

        if (not self.picked_key) and (self.carrying is not None) and (self.carrying.type == "key"):
            self.picked_key = True
            shaped += self.reward_cfg.reward_pickup_key

        if (not self.opened_door) and is_open1:
            self.opened_door = True
            shaped += self.reward_cfg.reward_open_door

        if terminated:
            shaped += self.reward_cfg.reward_goal

        # Penalise toggle only when door is already open or no door exists
        if action == self.actions.toggle and not (not was_open0 and is_open1):
            shaped += self.reward_cfg.pointless_toggle_penalty

        self._update_belief()

        total_reward = float(reward + shaped)

        info = dict(info)
        info["base_reward"] = float(reward)
        info["shaped_reward"] = float(shaped)
        info["total_reward"] = total_reward
        info.update(self._build_info())

        return obs, total_reward, terminated, truncated, info
    #registration 
def register_env():
    gym.register(
        id="SimpleKeyDoor-v0",
        entry_point="envs.simple_keydoor_env:NoisyBeliefKeyEnv",
    ) #aluete donce aluegt
    #ka Ppolaka dot add 
    #add a start 