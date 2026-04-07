"""
eval_greedy.py — greedy BFS baseline for SimpleKeyDoor.

The agent has full knowledge of the grid layout (cheating) and uses BFS
to find the shortest path to each sub-goal.  This gives a near-optimal
ceiling for navigation — PPO should aim to approach this performance.

Sub-goal sequence:
  1. Navigate adjacent to key  → face it → pickup
  2. Navigate adjacent to door → face it → toggle
  3. Navigate onto goal

Usage
-----
  python eval_greedy.py --episodes 100 --grid_size 15
  python eval_greedy.py --episodes 50 --noise 0.3 --n_lava 5
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import gymnasium as gym

from envs.simple_keydoor_env import register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper
from envs.pathfinding import optimal_path_length
from minigrid.wrappers import ImgObsWrapper


# ── Constants ─────────────────────────────────────────────────────────────────
# MiniGrid directions: 0=right, 1=down, 2=left, 3=up
DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
VEC_TO_DIR = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}

# ReducedActionSpace: 0=left, 1=right, 2=forward, 3=pickup, 4=toggle
ACT_LEFT    = 0
ACT_RIGHT   = 1
ACT_FORWARD = 2
ACT_PICKUP  = 3
ACT_TOGGLE  = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def _turn_action(current_dir: int, desired_dir: int) -> int:
    """Return ACT_LEFT or ACT_RIGHT to rotate toward desired_dir."""
    diff = (desired_dir - current_dir) % 4
    if diff == 0:
        return ACT_FORWARD  # already facing
    elif diff == 1:
        return ACT_RIGHT
    elif diff == 3:
        return ACT_LEFT
    else:  # diff == 2, 180°
        return ACT_RIGHT


def bfs_path(grid, start: tuple, goal: tuple, blocked: set | None = None) -> list[tuple]:
    """
    BFS shortest path from start to goal on the grid.
    Returns list of (x,y) cells from start to goal inclusive, or [] if unreachable.
    Walls and lava are impassable. Extra blocked positions can be specified.
    """
    if start == goal:
        return [start]

    blocked = blocked or set()
    w, h = grid.width, grid.height
    parent = {start: None}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in parent:
                continue
            if (nx, ny) in blocked:
                continue
            cell = grid.get(nx, ny)
            if cell is not None and cell.type in ("wall", "lava"):
                continue

            parent[(nx, ny)] = (x, y)

            if (nx, ny) == goal:
                path = []
                cur = (nx, ny)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            queue.append((nx, ny))

    return []


def bfs_adjacent(grid, start: tuple, target: tuple, blocked: set | None = None) -> list[tuple]:
    """
    BFS to find shortest path from start to any cell adjacent to target.
    Returns the path (start ... adjacent_cell) or [] if unreachable.
    The target cell itself is treated as blocked (can't walk onto it).
    """
    blocked = (blocked or set()) | {target}

    # Find all walkable cells adjacent to target
    w, h = grid.width, grid.height
    adj_goals = set()
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = target[0] + dx, target[1] + dy
        if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in blocked:
            cell = grid.get(nx, ny)
            if cell is None or cell.type not in ("wall", "lava"):
                adj_goals.add((nx, ny))

    if not adj_goals:
        return []

    if start in adj_goals:
        return [start]

    parent = {start: None}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in parent:
                continue
            if (nx, ny) in blocked:
                continue
            cell = grid.get(nx, ny)
            if cell is not None and cell.type in ("wall", "lava"):
                continue

            parent[(nx, ny)] = (x, y)

            if (nx, ny) in adj_goals:
                path = []
                cur = (nx, ny)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path

            queue.append((nx, ny))

    return []


# ── Agent ─────────────────────────────────────────────────────────────────────

class GreedyAgent:
    """
    BFS-based agent with full grid knowledge.

    State machine per sub-goal:
      NAVIGATE → follow BFS path to adjacent cell
      FACE     → turn to face the target
      INTERACT → pickup / toggle / step onto goal
    """

    def __init__(self):
        self.path: list[tuple] = []
        self.path_idx: int = 0
        self.state: str = "NAVIGATE"  # NAVIGATE, FACE, INTERACT

    def reset(self):
        self.path = []
        self.path_idx = 0
        self.state = "NAVIGATE"

    def act(self, base) -> int:
        pos = tuple(base.agent_pos)
        adir = base.agent_dir

        # ── Determine sub-goal ────────────────────────────────────────────
        has_key = base.carrying is not None and base.carrying.type == "key"
        door = base.grid.get(*base.door_pos)
        door_open = door is not None and door.type == "door" and door.is_open

        if not has_key:
            target = tuple(base.key_pos)
            interact_action = ACT_PICKUP
            need_adjacent = True
            blocked = {tuple(base.door_pos)}
        elif not door_open:
            target = tuple(base.door_pos)
            interact_action = ACT_TOGGLE
            need_adjacent = True
            blocked = set()
        else:
            target = tuple(base.goal_pos)
            interact_action = ACT_FORWARD
            need_adjacent = False  # walk onto the goal
            blocked = set()

        # ── Check if adjacent to target ───────────────────────────────────
        dx_t = target[0] - pos[0]
        dy_t = target[1] - pos[1]
        is_adjacent = abs(dx_t) + abs(dy_t) == 1

        if need_adjacent and is_adjacent:
            # Face the target and interact
            desired = VEC_TO_DIR.get((dx_t, dy_t))
            if desired is not None and adir != desired:
                return _turn_action(adir, desired)
            return interact_action

        # ── Check if ON the target (for goal) ─────────────────────────────
        if not need_adjacent and pos == target:
            return ACT_FORWARD  # shouldn't happen, env terminates

        # ── NAVIGATE: compute or follow BFS path ─────────────────────────
        # Recompute path if empty or we've drifted off it
        need_recompute = (
            not self.path
            or self.path_idx >= len(self.path)
            or pos != self.path[self.path_idx]
        )

        if need_recompute:
            if need_adjacent:
                self.path = bfs_adjacent(base.grid, pos, target, blocked)
            else:
                self.path = bfs_path(base.grid, pos, target, blocked)
            self.path_idx = 0

            if not self.path:
                # No path found — try random action as fallback
                return np.random.randint(0, 3)

        # Already at end of path?
        if self.path_idx + 1 >= len(self.path):
            # Should be adjacent or on target — handled above
            # Force recompute next step
            self.path = []
            return ACT_FORWARD

        # Move toward next cell in path
        next_cell = self.path[self.path_idx + 1]
        dx = next_cell[0] - pos[0]
        dy = next_cell[1] - pos[1]
        desired = VEC_TO_DIR.get((dx, dy))

        if desired is None:
            self.path = []
            return ACT_FORWARD

        if adir == desired:
            self.path_idx += 1
            return ACT_FORWARD
        else:
            return _turn_action(adir, desired)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Greedy BFS baseline for SimpleKeyDoor")
    parser.add_argument("--episodes",  type=int,   default=100)
    parser.add_argument("--grid_size", type=int,   default=15)
    parser.add_argument("--n_lava",    type=int,   default=0)
    parser.add_argument("--noise",     type=float, default=0.0,
                        help="Sensor noise probability (greedy agent ignores observations)")
    args = parser.parse_args()

    register_env()

    agent = GreedyAgent()
    ep_rewards = []
    ep_optimality = []
    successes = 0

    for ep in range(args.episodes):
        env = gym.make("SimpleKeyDoor-v0", render_mode=None,
                       size=args.grid_size, n_lava=args.n_lava)
        env = ReducedActionSpace(env)
        if args.noise > 0.0:
            env = SensorNoiseWrapper(env, noise_prob=args.noise)
        env = ImgObsWrapper(env)

        obs, info = env.reset()
        base = env.unwrapped
        agent.reset()

        opt_len = optimal_path_length(
            base.grid,
            tuple(base.agent_pos),
            tuple(base.key_pos),
            tuple(base.door_pos),
            tuple(base.goal_pos),
        )

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.act(base)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

        is_success = info.get("is_success", False)
        if is_success:
            successes += 1

        ep_rewards.append(total_reward)

        if opt_len > 0 and steps > 0:
            ratio = opt_len / steps
            ep_optimality.append(ratio)

        status = "SUCCESS" if is_success else "FAIL"
        print(f"Episode {ep + 1:>4}  steps={steps:>4}  reward={total_reward:>7.2f}  {status}")
        env.close()

    print(f"\n── Greedy BFS Baseline Summary ─────────────────────────────────────")
    print(f"Episodes:    {args.episodes}")
    print(f"Grid size:   {args.grid_size}")
    print(f"Noise:       {args.noise}")
    print(f"N lava:      {args.n_lava}")
    print(f"Success rate: {successes}/{args.episodes} = {successes / args.episodes:.2%}")
    print(f"Mean reward:  {np.mean(ep_rewards):.3f} | Std: {np.std(ep_rewards):.3f}")
    if ep_optimality:
        print(f"Mean path optimality: {np.mean(ep_optimality):.3f} | Std: {np.std(ep_optimality):.3f}")
    else:
        print("Mean path optimality: N/A (no successful episodes)")


if __name__ == "__main__":
    main()
