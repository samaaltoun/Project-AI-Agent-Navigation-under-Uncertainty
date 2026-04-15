"""
pathfinding.py — A* shortest-path computation for the SimpleKeyDoor environment.

Computes the optimal (minimum-cost) path to complete the key-door-goal task,
used as the denominator in the path optimality ratio metric.

Uses A* with Manhattan distance heuristic.  Supports weighted terrain so that
lava tiles can optionally have a finite traversal cost instead of being
completely impassable.
"""

from __future__ import annotations

import heapq


def astar_shortest(
    grid,
    start: tuple,
    goal: tuple,
    blocked_positions: set | None = None,
    lava_cost: float = float("inf"),
) -> int:
    """
    A* shortest path cost from *start* to *goal* on a MiniGrid Grid.

    Parameters
    ----------
    grid : minigrid.core.grid.Grid
        The environment's full grid object.
    start : tuple (x, y)
        Starting cell in world coordinates.
    goal : tuple (x, y)
        Target cell in world coordinates.
    blocked_positions : set of (x, y), optional
        Extra cells to treat as impassable (e.g. a locked door).
    lava_cost : float
        Cost to step onto a lava cell.
        Default ``inf`` → lava is impassable (same as a wall).
        Set to e.g. ``5.0`` to allow traversal at higher cost.

    Returns
    -------
    int
        Shortest path cost (number of steps when all costs are 1),
        or -1 if the goal is unreachable.
    """
    if start == goal:
        return 0

    blocked = blocked_positions or set()
    w, h = grid.width, grid.height

    def heuristic(pos: tuple) -> int:
        """Manhattan distance — admissible and consistent for 4-connected grids."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # Priority queue entries: (f_score, g_score, x, y)
    open_set = [(heuristic(start), 0, start[0], start[1])]
    g_scores: dict[tuple, float] = {start: 0}

    while open_set:
        _f, g, x, y = heapq.heappop(open_set)

        if (x, y) == goal:
            return int(g)

        # Skip stale entries
        if g > g_scores.get((x, y), float("inf")):
            continue

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in blocked:
                continue

            cell = grid.get(nx, ny)

            # Walls are always impassable
            if cell is not None and cell.type == "wall":
                continue

            # Lava has configurable cost
            if cell is not None and cell.type == "lava":
                if lava_cost == float("inf"):
                    continue
                step_cost = lava_cost
            else:
                step_cost = 1

            new_g = g + step_cost

            if new_g < g_scores.get((nx, ny), float("inf")):
                g_scores[(nx, ny)] = new_g
                f_score = new_g + heuristic((nx, ny))
                heapq.heappush(open_set, (f_score, new_g, nx, ny))

    return -1  # unreachable


def astar_on_belief(
    type_map: "np.ndarray",
    start: tuple,
    goal: tuple,
    blocked_positions: set | None = None,
    lava_cost: float = float("inf"),
) -> list[tuple[int, int]]:
    """
    A* shortest path on a 2-D **belief type map** (argmax of belief).

    Unlike ``astar_shortest`` which needs a real MiniGrid Grid object,
    this version works with a plain (W, H) numpy array where each cell
    contains the predicted object-type index (0-10).

    Object types treated as impassable:
        2 = wall   (always)
        9 = lava   (unless lava_cost < inf)

    Parameters
    ----------
    type_map : np.ndarray, shape (W, H)
        ``type_map[x, y]`` = most-likely MiniGrid object-type index.
    start, goal : tuple (x, y)
    blocked_positions : set of (x, y), optional
    lava_cost : float
        Cost to step onto a predicted-lava cell.

    Returns
    -------
    list of (x, y)
        The shortest path **including start and goal**.
        Empty list if unreachable.
    """
    import numpy as np          # local import to keep module light at top-level

    if start == goal:
        return [start]

    blocked = blocked_positions or set()
    W, H = type_map.shape

    WALL_TYPE = 2
    LAVA_TYPE = 9

    def heuristic(pos: tuple) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_set = [(heuristic(start), 0, start[0], start[1])]
    g_scores: dict[tuple, float] = {start: 0}
    came_from: dict[tuple, tuple] = {}

    while open_set:
        _f, g, x, y = heapq.heappop(open_set)

        if (x, y) == goal:
            # Reconstruct path
            path = [(x, y)]
            while (x, y) in came_from:
                x, y = came_from[(x, y)]
                path.append((x, y))
            path.reverse()
            return path

        if g > g_scores.get((x, y), float("inf")):
            continue

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if (nx, ny) in blocked:
                continue

            t = int(type_map[nx, ny])

            if t == WALL_TYPE:
                continue
            if t == LAVA_TYPE:
                if lava_cost == float("inf"):
                    continue
                step_cost = lava_cost
            else:
                step_cost = 1

            new_g = g + step_cost
            if new_g < g_scores.get((nx, ny), float("inf")):
                g_scores[(nx, ny)] = new_g
                came_from[(nx, ny)] = (x, y)
                heapq.heappush(open_set, (new_g + heuristic((nx, ny)), new_g, nx, ny))

    return []  # unreachable


def optimal_path_length(
    grid,
    agent_start: tuple,
    key_pos: tuple,
    door_pos: tuple,
    goal_pos: tuple,
    lava_cost: float = float("inf"),
) -> int:
    """
    Optimal cost for the full key → door → goal task using A*.

    The path has three segments:
      1. agent_start → key_pos   (door is locked, so treat it as blocked)
      2. key_pos     → door_pos  (door is still locked but we're walking TO it)
      3. door_pos    → goal_pos  (door is now open, passable)

    Parameters
    ----------
    lava_cost : float
        Passed through to ``astar_shortest`` for weighted terrain.

    Returns
    -------
    int
        Total optimal cost, or -1 if any segment is unreachable.
    """
    # Segment 1: start → key.  Door is locked = impassable.
    d1 = astar_shortest(grid, agent_start, key_pos,
                        blocked_positions={door_pos}, lava_cost=lava_cost)
    if d1 < 0:
        return -1

    # Segment 2: key → door.  Walking to the door cell itself.
    d2 = astar_shortest(grid, key_pos, door_pos,
                        blocked_positions=set(), lava_cost=lava_cost)
    if d2 < 0:
        return -1

    # Segment 3: door → goal.  Door is now open = passable.
    d3 = astar_shortest(grid, door_pos, goal_pos,
                        blocked_positions=set(), lava_cost=lava_cost)
    if d3 < 0:
        return -1

    return d1 + d2 + d3
