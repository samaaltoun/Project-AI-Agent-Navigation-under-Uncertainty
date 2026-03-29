# generate_level.py
# Generates a random level layout and saves it to a JSON file.
# The saved snapshot can then be loaded during training so every
# episode runs on the exact same grid.
#
# Usage examples
# --------------
# Basic (auto-named output):
#   python generate_level.py
#
# Specify seed + output path:
#   python generate_level.py --seed 42 --out levels/stage1.json
#
# Two-wall layout:
#   python generate_level.py --num_walls 2 --seed 7 --out levels/stage2.json
#
# In a training script:
#   from envs.variational_goal_env import load_snapshot, VariationalGoalEnv
#   snap = load_snapshot("levels/stage1.json")
#   env  = VariationalGoalEnv(snapshot=snap)   # same grid every reset

import argparse
import time
from pathlib import Path

from envs.variational_goal_env import LevelConfig, generate_snapshot, save_snapshot


def main():
    parser = argparse.ArgumentParser(description="Generate and save a fixed level layout.")
    parser.add_argument("--seed",         type=int,   default=None,       help="Random seed (omit for a truly random layout)")
    parser.add_argument("--width",        type=int,   default=20,         help="Grid width  (including border walls)")
    parser.add_argument("--height",       type=int,   default=16,         help="Grid height (including border walls)")
    parser.add_argument("--num_walls",    type=int,   default=1,          help="Number of dividing walls")
    parser.add_argument("--doors_per_wall", type=int, default=1,          help="Locked doors (and keys) per wall")
    parser.add_argument("--orientation",  type=str,   default="vertical", help="vertical | horizontal | random")
    parser.add_argument("--min_gap",      type=int,   default=3,          help="Minimum room width in cells")
    parser.add_argument("--out",          type=str,   default=None,       help="Output path (default: levels/<timestamp>.json)")
    args = parser.parse_args()

    level = LevelConfig(
        width            = args.width,
        height           = args.height,
        num_walls        = args.num_walls,
        doors_per_wall   = args.doors_per_wall,
        wall_orientation = args.orientation,
        min_gap          = args.min_gap,
    )

    print(f"Generating layout — walls={args.num_walls}, size={args.width}x{args.height}, seed={args.seed}")
    snap = generate_snapshot(level, seed=args.seed)

    if args.out is None:
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        out_path = Path("levels") / f"{ts}_w{args.num_walls}_seed{args.seed}.json"
    else:
        out_path = Path(args.out)

    save_snapshot(snap, out_path)

    print(f"\nLayout summary:")
    print(f"  Grid size      : {snap.width} x {snap.height}")
    print(f"  Walls          : {len(snap.wall_orientations)}  ({snap.wall_orientations})")
    print(f"  Wall coords    : {snap.wall_coords}")
    print(f"  Door positions : {snap.door_positions}")
    print(f"  Key  positions : {snap.key_positions}")
    print(f"  Goal position  : {snap.goal_pos}")
    print(f"  Agent start    : pos={snap.agent_start_pos}  dir={snap.agent_start_dir}")
    print(f"\nTo use in training:")
    print(f"  from envs.variational_goal_env import load_snapshot, VariationalGoalEnv")
    print(f"  snap = load_snapshot('{out_path}')")
    print(f"  env  = VariationalGoalEnv(snapshot=snap)")


if __name__ == "__main__":
    main()
