"""
pathfinding.py — BFS shortest-path computation for the SimpleKeyDoor environment.

Computes the optimal (minimum) number of steps to complete the key-door-goal
task under full observability, used as the denominator in the path optimality
ratio metric.
"""

from __future__ import annotations
from collections import deque


def bfs_shortest(grid, start: tuple, goal: tuple, blocked_positions: set | None = None) -> int:
    """
    BFS shortest distance (in grid steps) from start to goal on a MiniGrid Grid.

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

    Returns
    -------
    int
        Shortest path length in steps, or -1 if unreachable.
    """
    if start == goal:
        return 0

    blocked = blocked_positions or set()
    w, h = grid.width, grid.height
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in visited:
                continue
            if (nx, ny) in blocked:
                continue

            cell = grid.get(nx, ny)
            # Impassable: wall and lava
            if cell is not None and cell.type in ("wall", "lava"):
                continue

            if (nx, ny) == goal:
                return dist + 1

            visited.add((nx, ny))
            queue.append(((nx, ny), dist + 1))

    return -1  # unreachable


def optimal_path_length(grid, agent_start: tuple, key_pos: tuple,
                        door_pos: tuple, goal_pos: tuple) -> int:
    """
    Optimal steps for the full key → door → goal task.

    The path has three segments:
      1. agent_start → key_pos   (door is locked, so treat it as blocked)
      2. key_pos     → door_pos  (door is still locked but we're walking TO it)
      3. door_pos    → goal_pos  (door is now open, passable)

    Returns
    -------
    int
        Total optimal steps, or -1 if any segment is unreachable.
    """
    # Segment 1: start → key.  Door is locked = impassable.
    d1 = bfs_shortest(grid, agent_start, key_pos, blocked_positions={door_pos})
    if d1 < 0:
        return -1

    # Segment 2: key → door.  Walking to the door cell itself, so door_pos
    # is our goal, not a blocker.
    d2 = bfs_shortest(grid, key_pos, door_pos, blocked_positions=set())
    if d2 < 0:
        return -1

    # Segment 3: door → goal.  Door is now open = passable.
    d3 = bfs_shortest(grid, door_pos, goal_pos, blocked_positions=set())
    if d3 < 0:
        return -1

    return d1 + d2 + d3