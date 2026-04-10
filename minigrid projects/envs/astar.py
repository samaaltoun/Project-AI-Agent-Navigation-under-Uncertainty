"""
A* path-optimality helpers for the key-door navigation task.

These functions compute the theoretical shortest path an agent could
take with *full map visibility* (no fog, no sensor noise).  Comparing
this to the agent's actual path gives the Path Optimality Ratio — one
of the evaluation metrics required by the project specification.

Metric definition
-----------------
    Path Optimality Ratio = actual_cells_moved / optimal_a_star_cells

A ratio of 1.0 means the agent navigated perfectly.
A ratio of 2.0 means the agent walked twice the minimum distance.
The ratio is only meaningful for *successful* episodes.

Sub-goal decomposition for SimpleKeyDoorEnv
-------------------------------------------
The agent must:
  1.  Walk from start  → key       (locked door = impassable wall)
  2.  Walk from key    → goal      (door treated as open / passable)

  optimal = d1 + d2

Usage
-----
    from envs.astar import optimal_path_length, build_passable

    # Capture once per episode, right after env.reset():
    opt = optimal_path_length(
        base.grid, grid_size,
        start=tuple(base.agent_pos),
        key_pos=tuple(base.key_pos),
        door_pos=tuple(base.door_pos),
        goal_pos=tuple(base.goal_pos),
    )

    # Count how many cells the agent actually moved:
    #   increment actual_cells whenever base.agent_pos changes each step.

    ratio = actual_cells / opt   # only when is_success and opt is not None
"""

from __future__ import annotations

import heapq


# ── Core A* ───────────────────────────────────────────────────────────────────

def astar(passable: set[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int]) -> int | None:
    """
    Shortest path length (number of cell moves) from *start* to *goal* using A*.

    Parameters
    ----------
    passable : set of (x, y)
        All grid cells the agent is allowed to walk through.
        The *goal* cell itself is always reachable regardless of membership.
    start, goal : (x, y) tuples

    Returns
    -------
    int
        Number of cell-to-cell moves on the shortest path.
    None
        If no path exists (start and goal are disconnected).

    How A* works (conceptually)
    ---------------------------
    A* is like Dijkstra's shortest-path search, but guided by a heuristic
    that estimates the remaining distance.  Here we use the Manhattan distance
    |Δx| + |Δy|, which is *admissible* (never over-estimates) on a grid where
    only 4-directional movement is allowed.

    Each node in the open set is (f = g + h, g, position):
      g = actual cost from start to this cell
      h = heuristic estimate to goal
      f = estimated total cost

    We always expand the node with the smallest f first.  Once we pop the
    goal, g is guaranteed to be optimal.
    """
    if start == goal:
        return 0

    def h(pos: tuple[int, int]) -> int:
        """Manhattan distance heuristic — admissible for 4-connected grids."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # Min-heap: (f_score, g_score, position)
    open_heap: list = [(h(start), 0, start)]
    g_score: dict[tuple[int, int], int] = {start: 0}

    while open_heap:
        _, cost, cur = heapq.heappop(open_heap)

        if cur == goal:
            return cost

        # Skip stale entries (a shorter path to cur was already found)
        if cost > g_score.get(cur, float("inf")):
            continue

        # Expand 4-connected neighbours
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cur[0] + dx, cur[1] + dy)
            # Goal is always enterable; all other cells must be in passable
            if nb != goal and nb not in passable:
                continue
            new_cost = cost + 1
            if new_cost < g_score.get(nb, float("inf")):
                g_score[nb] = new_cost
                heapq.heappush(open_heap, (new_cost + h(nb), new_cost, nb))

    return None  # no path found


# ── Grid helpers ──────────────────────────────────────────────────────────────

def build_passable(
    grid,
    grid_size: int,
    extra: tuple[tuple[int, int], ...] = (),
) -> set[tuple[int, int]]:
    """
    Extract the set of walkable cells from a MiniGrid ``Grid`` object.

    Impassable cell types: ``wall``, ``lava``, ``door``
    (locked doors are typed "door" in MiniGrid and block movement).

    Parameters
    ----------
    grid : minigrid.core.grid.Grid
        The ground-truth grid (e.g. ``env.unwrapped.grid``).
        Read this **immediately after** ``env.reset()`` — before the agent
        takes any steps — so the layout reflects the initial state.
    grid_size : int
        Side length of the square grid.
    extra : tuple of (x, y)
        Additional positions to force-include as passable (e.g. the door
        cell when computing a path segment where the door is assumed open).

    Returns
    -------
    set of (x, y)
        All walkable cell coordinates.
    """
    blocked = {"wall", "lava", "door"}
    passable: set[tuple[int, int]] = set()
    for x in range(grid_size):
        for y in range(grid_size):
            cell = grid.get(x, y)
            if cell is None or cell.type not in blocked:
                passable.add((x, y))
    passable.update(extra)
    return passable


# ── High-level metric ─────────────────────────────────────────────────────────

def optimal_path_length(
    grid,
    grid_size: int,
    start: tuple[int, int],
    key_pos: tuple[int, int],
    door_pos: tuple[int, int],
    goal_pos: tuple[int, int],
) -> int | None:
    """
    Compute the *minimum* number of cell moves to complete the key-door task,
    assuming full map visibility (no sensor noise, no fog of war).

    Sub-goal decomposition
    ----------------------
    Segment 1:  start  →  key
        The door is locked (impassable).  The agent must reach the key first.

    Segment 2:  key  →  goal
        The door is now treated as open (passable).  The agent walks straight
        to the goal through the wall opening.

    Total optimal = d1 + d2

    Why this decomposition?
    -----------------------
    The agent *must* visit the key before it can open the door.  These two
    A* segments give a lower bound on the actual distance any correct agent
    must travel.  Comparing the agent's actual cell-move count to this lower
    bound produces the Path Optimality Ratio.

    Parameters
    ----------
    grid : minigrid.core.grid.Grid
        Ground-truth grid, captured right after env.reset().
    grid_size : int
        Side length of the square grid.
    start : (x, y)
        Agent's position at the start of the episode.
    key_pos : (x, y)
        Position of the key object.
    door_pos : (x, y)
        Position of the locked door.
    goal_pos : (x, y)
        Position of the goal tile.

    Returns
    -------
    int
        Minimum cell-move count.
    None
        If either segment is unreachable (degenerate map — log and skip).
    """
    # Segment 1: door locked → agent cannot cross the wall
    passable_locked = build_passable(grid, grid_size)

    # Segment 2: door open → add door cell to passable set
    passable_open = build_passable(grid, grid_size, extra=(door_pos,))

    d1 = astar(passable_locked, start, key_pos)
    if d1 is None:
        return None  # key is unreachable — should never happen in a valid map

    d2 = astar(passable_open, key_pos, goal_pos)
    if d2 is None:
        return None  # goal is unreachable — should never happen

    return d1 + d2
