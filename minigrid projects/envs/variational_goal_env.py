# envs/variational_goal_env.py
# This is the variational goal environment used for training the PPO agent in train_ppo_var.py,
# it has parameters that can be varied such as number of walls, doors per wall, wall orientation, and grid dimensions


from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

# One colour per door/key pair — supports up to 6 pairs before cycling
_PAIR_COLOURS = ["yellow", "red", "blue", "green", "purple", "grey"]


@dataclass
class ShapingConfig:
    reward_pickup_key: float = 2.0
    reward_open_door: float = 1.5
    reward_goal: float = 5.0
    step_penalty: float = -0.001


@dataclass
class LevelConfig:
    """
    Controls procedural generation each episode.

    width / height      : total grid size including border walls (min 7 each)
    num_walls           : number of internal dividing walls
    doors_per_wall      : locked doors (and matching keys) per wall
    wall_orientation    : 'vertical', 'horizontal', or 'random'
    min_gap             : minimum interior cells between consecutive parallel walls
                          (prevents rooms too narrow to navigate)

    Examples
    --------
    Single room, classic layout (default):
        LevelConfig(width=10, height=10, num_walls=1, doors_per_wall=1)

    Wide asymmetric corridor, two keys per door:
        LevelConfig(width=14, height=7, num_walls=1, doors_per_wall=2)

    Three-room maze with one door each:
        LevelConfig(width=16, height=10, num_walls=2, doors_per_wall=1)

    Mixed-orientation grid:
        LevelConfig(width=14, height=14, num_walls=2, wall_orientation='random')
    """
    width: int = 20
    height: int = 16
    num_walls: int = 2
    doors_per_wall: int = 1
    wall_orientation: str = "vertical"   # 'vertical' | 'horizontal' | 'random'
    min_gap: int = 3


# -----------------------------------------------------------------------
# Level snapshot — a fully-specified, reproducible layout
# -----------------------------------------------------------------------

@dataclass
class LevelSnapshot:
    """
    Stores every position decision made during one procedural generation so
    the exact same grid can be reconstructed on every subsequent reset.

    Fields
    ------
    width / height          : grid dimensions (including border walls)
    wall_orientations       : one entry per wall — 'vertical' or 'horizontal'
    wall_coords             : the x (vertical) or y (horizontal) coordinate of
                              each wall, same order as wall_orientations
    door_positions          : flat list of (x, y) for every locked door
    door_colours            : colour string for each door (matches door_positions)
    key_positions           : flat list of (x, y) for every key (same order as
                              door_positions — each key unlocks its matching door)
    goal_pos                : (x, y) of the Goal tile
    agent_start_pos         : (x, y) where the agent spawns each episode
    agent_start_dir         : facing direction  0=right 1=down 2=left 3=up
    seed                    : the numpy seed used to generate this layout
                              (stored for documentation; not re-applied automatically)
    """
    width: int
    height: int
    wall_orientations: List[str]
    wall_coords: List[int]
    door_positions: List[Tuple[int, int]]
    door_colours: List[str]
    key_positions: List[Tuple[int, int]]
    goal_pos: Tuple[int, int]
    agent_start_pos: Tuple[int, int]
    agent_start_dir: int
    seed: Optional[int] = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "width":             self.width,
            "height":            self.height,
            "wall_orientations": self.wall_orientations,
            "wall_coords":       self.wall_coords,
            "door_positions":    [list(p) for p in self.door_positions],
            "door_colours":      self.door_colours,
            "key_positions":     [list(p) for p in self.key_positions],
            "goal_pos":          list(self.goal_pos),
            "agent_start_pos":   list(self.agent_start_pos),
            "agent_start_dir":   self.agent_start_dir,
            "seed":              self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LevelSnapshot":
        return cls(
            width             = d["width"],
            height            = d["height"],
            wall_orientations = d["wall_orientations"],
            wall_coords       = d["wall_coords"],
            door_positions    = [tuple(p) for p in d["door_positions"]],
            door_colours      = d["door_colours"],
            key_positions     = [tuple(p) for p in d["key_positions"]],
            goal_pos          = tuple(d["goal_pos"]),
            agent_start_pos   = tuple(d["agent_start_pos"]),
            agent_start_dir   = d["agent_start_dir"],
            seed              = d.get("seed"),
        )


def save_snapshot(snapshot: LevelSnapshot, path: str | Path) -> None:
    """Write a LevelSnapshot to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved level snapshot → {path}")


def load_snapshot(path: str | Path) -> LevelSnapshot:
    """Load a LevelSnapshot from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LevelSnapshot.from_dict(data)


def generate_snapshot(level: LevelConfig, seed: Optional[int] = None) -> LevelSnapshot:
    """
    Randomly generate one layout for the given LevelConfig and capture it as a
    LevelSnapshot.  Passing the same seed always produces the same snapshot.

    Usage
    -----
    snap = generate_snapshot(LevelConfig(num_walls=1), seed=42)
    save_snapshot(snap, "levels/stage1.json")

    # Later, during training:
    snap = load_snapshot("levels/stage1.json")
    env  = VariationalGoalEnv(snapshot=snap)   # same grid every episode
    """
    if seed is not None:
        np.random.seed(seed)

    # Spin up a temporary env purely to run _gen_grid once and harvest positions.
    tmp = VariationalGoalEnv(level=level, render_mode=None)
    tmp.reset()

    snap = LevelSnapshot(
        width             = level.width,
        height            = level.height,
        wall_orientations = list(tmp._snap_wall_orientations),
        wall_coords       = list(tmp._snap_wall_coords),
        door_positions    = [tuple(p) for p in tmp.door_positions],
        door_colours      = list(tmp._snap_door_colours),
        key_positions     = [tuple(p) for p in tmp.key_positions],
        goal_pos          = tuple(tmp.goal_pos),
        agent_start_pos   = tuple(int(x) for x in tmp.agent_pos),
        agent_start_dir   = int(tmp.agent_dir),
        seed              = seed,
    )
    tmp.close()
    return snap


# -----------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------

def sample_unique(region: list, forbidden: set) -> tuple:
    """Random position from region not in forbidden."""
    candidates = [p for p in region if p not in forbidden]
    if not candidates:
        raise ValueError("No valid positions available in region.")
    return candidates[int(np.random.randint(len(candidates)))]


def _spread_coords(n: int, lo: int, hi: int, min_gap: int) -> List[int]:
    """
    Return n randomly-placed integer coordinates inside [lo, hi]
    with at least min_gap between consecutive values.
    """
    if n == 0:
        return []
    span_needed = (n - 1) * min_gap
    if hi - lo < span_needed:
        raise ValueError(
            f"Grid too small for {n} walls with min_gap={min_gap} "
            f"in range [{lo}, {hi}]. Increase grid size or reduce num_walls/min_gap."
        )
    # Iteratively sample, removing coords too close to already chosen ones
    available = list(range(lo, hi + 1))
    coords = []
    for _ in range(n):
        if not available:
            break
        c = available[int(np.random.randint(len(available)))]
        coords.append(c)
        available = [x for x in available if abs(x - c) >= min_gap]
    coords.sort()
    return coords


def _build_rooms(
    width: int, height: int, v_splits: List[int], h_splits: List[int]
) -> List[Tuple[int, int, int, int]]:
    """
    Return (x_lo, y_lo, x_hi, y_hi) bounding boxes for every room,
    ordered left-to-right, then top-to-bottom.
    Coordinates are inclusive interior-cell boundaries.
    """
    xs = [1] + [s + 1 for s in v_splits] + [width - 1]
    ys = [1] + [s + 1 for s in h_splits] + [height - 1]
    rooms = []
    for xi in range(len(xs) - 1):
        for yi in range(len(ys) - 1):
            rooms.append((xs[xi], ys[yi], xs[xi + 1], ys[yi + 1]))
    return rooms


def _room_cells(room: Tuple[int, int, int, int]) -> List[tuple]:
    x_lo, y_lo, x_hi, y_hi = room
    return [(x, y) for x in range(x_lo, x_hi) for y in range(y_lo, y_hi)]


def _cells_left_of(wall_x: int, height: int, v_splits: List[int]) -> List[tuple]:
    """Interior cells strictly to the left of wall_x, within its slab."""
    prev = max((s for s in v_splits if s < wall_x), default=0)
    x_lo = prev + 1 if prev > 0 else 1
    return [(x, y) for x in range(x_lo, wall_x) for y in range(1, height - 1)]


def _cells_above_wall(wall_y: int, width: int, h_splits: List[int]) -> List[tuple]:
    """Interior cells strictly above wall_y, within its slab."""
    prev = max((s for s in h_splits if s < wall_y), default=0)
    y_lo = prev + 1 if prev > 0 else 1
    return [(x, y) for x in range(1, width - 1) for y in range(y_lo, wall_y)]


def _place_wall_with_doors(
    grid: Grid,
    orientation: str,
    coord: int,
    width: int,
    height: int,
    num_doors: int,
    colour: str,
    forbidden: set,
) -> Tuple[List[tuple], List[tuple]]:
    """
    Draw a full dividing wall then punch num_doors locked doors into it.
    Returns (all_wall_cells, door_positions).
    """
    wall_cells = []
    if orientation == "vertical":
        for y in range(1, height - 1):
            grid.set(coord, y, Wall())
            wall_cells.append((coord, y))
    else:
        for x in range(1, width - 1):
            grid.set(x, coord, Wall())
            wall_cells.append((x, coord))

    # Punch doors — keep a local forbidden set so doors aren't adjacent
    door_positions = []
    door_forbidden = set(forbidden)
    for _ in range(num_doors):
        candidates = [p for p in wall_cells if p not in door_forbidden]
        if not candidates:
            break
        pos = candidates[int(np.random.randint(len(candidates)))]
        grid.set(pos[0], pos[1], Door(colour, is_locked=True))
        door_positions.append(pos)
        door_forbidden.add(pos)
        # Buffer so consecutive doors aren't adjacent
        if orientation == "vertical":
            door_forbidden.update([(pos[0], pos[1] - 1), (pos[0], pos[1] + 1)])
        else:
            door_forbidden.update([(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1])])

    return wall_cells, door_positions


# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------

class VariationalGoalEnv(MiniGridEnv):
    """
    Procedurally generated key-door grid environment.

    Pass a LevelConfig to control grid dimensions, number of walls,
    doors per wall, wall orientation, and minimum room width.

    Usage
    -----
    # Default (10x10, one vertical wall, one door/key)
    env = VariationalGoalEnv()

    # Asymmetric grid, two walls, two doors each
    env = VariationalGoalEnv(level=LevelConfig(
        width=16, height=8, num_walls=2, doors_per_wall=2
    ))

    # Three-room maze, random orientations
    env = VariationalGoalEnv(level=LevelConfig(
        width=14, height=14, num_walls=2, wall_orientation='random'
    ))
    """

    def __init__(
        self,
        level: LevelConfig | None = None,
        snapshot: LevelSnapshot | None = None,
        max_steps: int | None = None,
        shaping: ShapingConfig | None = None,
        **kwargs,
    ):
        self.level = level or LevelConfig()
        self.shaping = shaping or ShapingConfig()

        # When a snapshot is provided every reset rebuilds the exact same grid.
        # When None the env generates a fresh random layout each episode (default).
        self._snapshot = snapshot

        # If snapshot provided, override level dimensions to match
        if snapshot is not None:
            self.level = LevelConfig(
                width=snapshot.width,
                height=snapshot.height,
                num_walls=len(snapshot.wall_orientations),
                doors_per_wall=max(1, len(snapshot.door_positions) // max(len(snapshot.wall_orientations), 1)),
                wall_orientation=snapshot.wall_orientations[0] if snapshot.wall_orientations else "vertical",
            )

        # Populated each episode by _gen_grid
        self.door_positions: List[tuple] = []
        self.key_positions: List[tuple] = []
        self.goal_pos: tuple = (1, 1)
        self._keys_picked: List[bool] = []
        self._doors_opened: List[bool] = []

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * self.level.width * self.level.height

        super().__init__(
            mission_space=mission_space,
            width=self.level.width,
            height=self.level.height,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Collect all keys, open all doors, and reach the goal."

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        n = len(self.door_positions)
        self._keys_picked = [False] * n
        self._doors_opened = [False] * n
        self._visited: set = set()
        self._steps_no_new_cell = 0
        return obs, info

    # ------------------------------------------------------------------

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # ------------------------------------------------------------------
        # SNAPSHOT MODE — replay a fixed, pre-generated layout
        # ------------------------------------------------------------------
        if self._snapshot is not None:
            snap = self._snapshot

            # Rebuild walls (full lines first)
            for orientation, coord in zip(snap.wall_orientations, snap.wall_coords):
                if orientation == "vertical":
                    for y in range(1, height - 1):
                        self.grid.set(coord, y, Wall())
                else:
                    for x in range(1, width - 1):
                        self.grid.set(x, coord, Wall())

            # Punch in locked doors
            for pos, colour in zip(snap.door_positions, snap.door_colours):
                self.grid.set(pos[0], pos[1], Door(colour, is_locked=True))

            # Place matching keys
            for pos, colour in zip(snap.key_positions, snap.door_colours):
                self.grid.set(pos[0], pos[1], Key(colour))

            # Place goal
            self.grid.set(snap.goal_pos[0], snap.goal_pos[1], Goal())

            # Place agent (same start every episode)
            self.agent_pos = np.array(snap.agent_start_pos)
            self.agent_dir = snap.agent_start_dir

            # Sync tracked state
            self.door_positions = list(snap.door_positions)
            self.key_positions  = list(snap.key_positions)
            self.goal_pos       = snap.goal_pos

            n = len(self.door_positions)
            self._keys_picked        = [False] * n
            self._doors_opened       = [False] * n
            self._visited            = {tuple(self.agent_pos)}
            self._steps_no_new_cell  = 0
            self.mission             = self._gen_mission()
            return

        # ------------------------------------------------------------------
        # RANDOM MODE — generate a fresh layout (original behaviour)
        # ------------------------------------------------------------------
        cfg = self.level
        forbidden: set = set()

        # --- Decide orientations ---
        if cfg.wall_orientation == "random":
            orientations = [
                str(np.random.choice(["vertical", "horizontal"]))
                for _ in range(cfg.num_walls)
            ]
        else:
            orientations = [cfg.wall_orientation] * cfg.num_walls

        # --- Compute wall coordinates ---
        v_wall_indices = [i for i, o in enumerate(orientations) if o == "vertical"]
        h_wall_indices = [i for i, o in enumerate(orientations) if o == "horizontal"]

        wall_coords: Dict[int, int] = {}

        if v_wall_indices:
            coords = _spread_coords(
                n=len(v_wall_indices),
                lo=1 + cfg.min_gap,
                hi=width - 2 - cfg.min_gap,
                min_gap=cfg.min_gap,
            )
            for idx, c in zip(v_wall_indices, coords):
                wall_coords[idx] = c

        if h_wall_indices:
            coords = _spread_coords(
                n=len(h_wall_indices),
                lo=1 + cfg.min_gap,
                hi=height - 2 - cfg.min_gap,
                min_gap=cfg.min_gap,
            )
            for idx, c in zip(h_wall_indices, coords):
                wall_coords[idx] = c

        v_splits = sorted(wall_coords[i] for i in v_wall_indices)
        h_splits = sorted(wall_coords[i] for i in h_wall_indices)

        # --- Place walls + doors + keys ---
        all_door_positions: List[tuple] = []
        all_key_positions: List[tuple] = []

        for wall_idx in range(cfg.num_walls):
            orientation = orientations[wall_idx]
            coord = wall_coords[wall_idx]
            colour = _PAIR_COLOURS[wall_idx % len(_PAIR_COLOURS)]

            _, door_positions = _place_wall_with_doors(
                grid=self.grid,
                orientation=orientation,
                coord=coord,
                width=width,
                height=height,
                num_doors=cfg.doors_per_wall,
                colour=colour,
                forbidden=forbidden,
            )

            all_door_positions.extend(door_positions)
            forbidden.update(door_positions)

            # Key for each door goes in the room on the agent's approach side
            for door_pos in door_positions:
                if orientation == "vertical":
                    key_region = _cells_left_of(coord, height, v_splits)
                else:
                    key_region = _cells_above_wall(coord, width, h_splits)

                key_pos = sample_unique(key_region, forbidden)
                self.grid.set(key_pos[0], key_pos[1], Key(colour))
                all_key_positions.append(key_pos)
                forbidden.add(key_pos)

        self.door_positions = all_door_positions
        self.key_positions = all_key_positions

        # --- Rooms ---
        rooms = _build_rooms(width, height, v_splits, h_splits)

        # Goal in the last room (furthest from start)
        goal_region = _room_cells(rooms[-1])
        self.goal_pos = sample_unique(goal_region, forbidden)
        self.grid.set(self.goal_pos[0], self.goal_pos[1], Goal())
        forbidden.add(self.goal_pos)

        # Agent in the first room — use MiniGrid's place_agent so the
        # can_overlap assertion is never triggered
        x_lo, y_lo, x_hi, y_hi = rooms[0]
        self.place_agent(
            top=(x_lo, y_lo),
            size=(x_hi - x_lo, y_hi - y_lo),
        )

        self.mission = self._gen_mission()

        # Sync tracking lists
        n = len(self.door_positions)
        self._keys_picked = [False] * n
        self._doors_opened = [False] * n

        self._visited: set = {tuple(self.agent_pos)}
        self._steps_no_new_cell: int = 0

    # ------------------------------------------------------------------

    def step(self, action: int):
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def next_target() -> tuple:
            """Sub-goal: matching door if carrying key → next unpicked key → next unopened door → goal."""
            if self.carrying is not None and self.carrying.type == "key":
                key_colour = self.carrying.color
                for i, colour in enumerate(_PAIR_COLOURS[:len(self.door_positions)]):
                    if colour == key_colour and not self._doors_opened[i]:
                        return self.door_positions[i]
            for i, picked in enumerate(self._keys_picked):
                if not picked:
                    return self.key_positions[i]
            for i, opened in enumerate(self._doors_opened):
                if not opened:
                    return self.door_positions[i]
            return self.goal_pos

        gamma = 0.99
        k = 0.15

        # --- PRE-STEP ---
        pos0 = tuple(self.agent_pos)
        phi0 = -manhattan(pos0, next_target())

        fwd0 = tuple(self.front_pos)
        obj_ahead0 = self.grid.get(fwd0[0], fwd0[1])
        ahead_is_locked_door = (
            obj_ahead0 is not None
            and obj_ahead0.type == "door"
            and obj_ahead0.is_locked
        )
        has_key = self.carrying is not None and self.carrying.type == "key"
        was_open = [self._door_is_open(dp) for dp in self.door_positions]

        shaped = float(self.shaping.step_penalty)

        # --- TRANSITION ---
        obs, reward, terminated, truncated, info = super().step(action)
        pos1 = tuple(self.agent_pos)

        if pos1 not in self._visited:
            self._visited.add(pos1)
            shaped += 0.05  # increased bonus for new cells
            self._steps_no_new_cell = 0
        else:
            self._steps_no_new_cell += 1
            if self._steps_no_new_cell > 15:
                shaped -= 0.01 * min(self._steps_no_new_cell / 15, 2.0)

        # Bump penalty
        if action == self.actions.forward and pos1 == pos0:
            shaped -= 0.05
        # bonus for successful forward movement
        elif action == self.actions.forward and pos1 != pos0:
            shaped += 0.02

        # penalty for doing nothing while carrying a key
        if self.carrying is not None and pos1 == pos0:
            shaped -= 0.02

        # adjacency reward — carrying key and close to matching door
        if self.carrying is not None and self.carrying.type == "key":
            key_colour = self.carrying.color
            for i, colour in enumerate(_PAIR_COLOURS[:len(self.door_positions)]):
                if colour == key_colour and not self._doors_opened[i]:
                    dist_to_door = manhattan(pos1, self.door_positions[i])
                    if dist_to_door == 1:
                        shaped += 0.05
                    if dist_to_door == 0:
                        shaped += 0.1
                    break

        # --- EVENT REWARDS ---
        # Key pickup — only fires once per key
        if has_key:
            for i, picked in enumerate(self._keys_picked):
                if not picked:
                    # Match by colour if possible
                    key_colour = self.carrying.color
                    expected_colour = _PAIR_COLOURS[i % len(_PAIR_COLOURS)]
                    if key_colour == expected_colour or True:
                        self._keys_picked[i] = True
                        shaped += self.shaping.reward_pickup_key
                        break

        # Door opened
        opened_this_step = False
        for i, (opened, was) in enumerate(zip(self._doors_opened, was_open)):
            if not opened and self._door_is_open(self.door_positions[i]) and not was:
                self._doors_opened[i] = True
                shaped += self.shaping.reward_open_door
                opened_this_step = True

        # Goal reached
        if terminated:
            shaped += self.shaping.reward_goal

        # --- POTENTIAL SHAPING ---
        phi1 = -manhattan(pos1, next_target())
        # stronger pull toward goal once all doors are open
        all_doors_open = all(self._doors_opened) and len(self._doors_opened) > 0
        k_current = 0.3 if all_doors_open else k
        shaped += k_current * (gamma * phi1 - phi0)

        # --- DOOR BOTTLENECK ---
        if ahead_is_locked_door and has_key:
            if action == self.actions.toggle:
                shaped += 0.3 if opened_this_step else -0.05
            elif action == self.actions.forward and pos1 == pos0:
                shaped -= 0.05

        if action == self.actions.toggle and not opened_this_step and not (ahead_is_locked_door and has_key):
            shaped -= 0.02

        if pos1 == pos0 and action in (self.actions.left, self.actions.right, self.actions.toggle):
            shaped -= 0.05

        total_reward = float(reward + shaped)

        info = dict(info)
        info["shaped_reward"] = shaped
        info["keys_picked"] = list(self._keys_picked)
        info["doors_opened"] = list(self._doors_opened)
        info["agent_pos"] = tuple(self.agent_pos)
        info["goal_pos"] = tuple(self.goal_pos)
        info["is_success"] = bool(terminated and tuple(self.agent_pos) == tuple(self.goal_pos))

        return obs, total_reward, terminated, truncated, info

    def _door_is_open(self, pos: tuple) -> bool:
        obj = self.grid.get(pos[0], pos[1])
        return obj is not None and obj.type == "door" and obj.is_open


# -----------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------

def register_env():
    env_id = "VariationalGoal-v0"
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point="envs.variational_goal_env:VariationalGoalEnv",
        )
