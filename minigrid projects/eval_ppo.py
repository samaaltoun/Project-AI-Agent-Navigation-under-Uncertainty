import argparse
import json
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from minigrid.wrappers import ImgObsWrapper

from envs.simple_keydoor_env import register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper, BeliefObsWrapper
from envs.extractors import MiniGridCNN

from gymnasium.wrappers import RecordVideo

from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # file-only backend; safe on headless machines
import matplotlib.pyplot as plt
from envs.belief_map import BeliefMap
from envs.pathfinding import optimal_path_length


def _plot_belief_metrics(
    entropy_histories: list,
    kl_histories: list,
    noise_prob: float,
    model_path: str,
) -> None:
    max_len = max(len(h) for h in entropy_histories)

    def _pad(hists):
        return np.array([h + [h[-1]] * (max_len - len(h)) for h in hists], dtype=float)

    mean_h  = _pad(entropy_histories).mean(axis=0)
    std_h   = _pad(entropy_histories).std(axis=0)
    mean_kl = _pad(kl_histories).mean(axis=0)
    std_kl  = _pad(kl_histories).std(axis=0)
    steps   = np.arange(max_len)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(steps, mean_h, color="steelblue", linewidth=1.5)
    axes[0].fill_between(steps, mean_h - std_h, mean_h + std_h,
                         alpha=0.25, color="steelblue", label="±1 std")
    axes[0].set_xlabel("Step within episode")
    axes[0].set_ylabel("Mean entropy (bits)")
    axes[0].set_title(f"Belief entropy over time  (noise={noise_prob})")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linewidth=0.5, alpha=0.6)

    axes[1].plot(steps, mean_kl, color="coral", linewidth=1.5)
    axes[1].fill_between(steps, mean_kl - std_kl, mean_kl + std_kl,
                         alpha=0.25, color="coral", label="±1 std")
    axes[1].set_xlabel("Step within episode")
    axes[1].set_ylabel("KL divergence (bits)")
    axes[1].set_title(f"Belief vs ground truth KL  (noise={noise_prob})")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linewidth=0.5, alpha=0.6)

    plt.suptitle(
        f"Belief map — {len(entropy_histories)} episodes, noise_prob={noise_prob}",
        fontsize=11,
    )
    plt.tight_layout()

    out_path = Path(model_path).parent / f"belief_entropy_kl_noise{noise_prob}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved belief metrics plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", action="store_true", help="Save videos to disk")
    parser.add_argument("--video_dir", type=str, default="videos")
    parser.add_argument("--video_prefix", type=str, default="ppo_simplekeydoor")
    parser.add_argument("--noise", type=float, default=None, help="Sensor noise probability")
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--n_lava", type=int, default=None)
    parser.add_argument("--belief", action="store_true",
                        help="Track belief map per step; save entropy/KL plot next to model")
    parser.add_argument("--only_success", action="store_true")
    args = parser.parse_args()

    # Auto-load config from training run
    config_path = Path(args.model).parent / "config.json"
    saved = {}
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))

    if args.grid_size is None:
        args.grid_size = saved.get("grid_size", 15)
    if args.n_lava is None:
        args.n_lava = saved.get("n_lava", 0)
    if args.noise is None:
        args.noise = saved.get("noise_prob", 0.0)

    decay = saved.get("decay", 0.0)
    tpr = saved.get("tpr", None)

    register_env()

    # Detect whether the model was trained with BeliefObsWrapper
    use_belief_obs = "decay" in saved

    def make_eval_env():
        render_mode = "rgb_array" if args.video else ("human" if args.render else None)
        e = gym.make("SimpleKeyDoor-v0", render_mode=render_mode,
                     size=args.grid_size, n_lava=args.n_lava)
        e = ReducedActionSpace(e)
        if args.noise > 0.0:
            e = SensorNoiseWrapper(e, noise_prob=args.noise)

        if use_belief_obs:
            e = BeliefObsWrapper(e, grid_size=args.grid_size,
                                 noise_prob=args.noise, tpr=tpr, decay=decay)
        else:
            e = ImgObsWrapper(e)

        if args.video:
            e = RecordVideo(e, video_folder=args.video_dir,
                            episode_trigger=lambda ep_id: True,
                            name_prefix=args.video_prefix, disable_logger=True)
        return e

    env = DummyVecEnv([make_eval_env])
    if not use_belief_obs:
        env = VecTransposeImage(env)  # BeliefObsWrapper already outputs (C,H,W)

    model = PPO.load(args.model, env=env,
                     custom_objects={"features_extractor_class": MiniGridCNN})

    base = env.venv.envs[0].unwrapped

    # Access belief map from wrapper (if using belief obs) or create standalone
    belief_wrapper = None
    if use_belief_obs:
        # Walk the wrapper chain to find BeliefObsWrapper
        w = env.venv.envs[0]
        while w is not None:
            if isinstance(w, BeliefObsWrapper):
                belief_wrapper = w
                break
            w = getattr(w, "env", None)

    bmap = belief_wrapper.bmap if belief_wrapper else (
        BeliefMap(grid_size=args.grid_size, noise_prob=args.noise) if args.belief else None
    )

    all_entropy_histories: list = []
    all_kl_histories: list = []

    ep_rewards = []
    ep_optimality = []
    successes  = 0
    ep_shown   = 0
    ep_attempted = 0
    max_attempts = args.episodes * 20

    while ep_shown < args.episodes:
        if args.only_success and ep_attempted >= max_attempts:
            print(f"Warning: hit attempt limit ({max_attempts}).")
            break

        obs = env.reset()
        optimal_len = optimal_path_length(
            base.grid, tuple(base.agent_pos), tuple(base.key_pos),
            tuple(base.door_pos), tuple(base.goal_pos),
        )
        done  = False
        total = 0.0
        step  = 0

        # Belief tracking for standalone bmap (non-wrapper mode)
        if bmap is not None and belief_wrapper is None:
            bmap.reset()
            raw_obs = env.venv.envs[0].unwrapped
            # For standalone bmap, we can't easily get raw obs — skip per-step tracking

        if bmap is not None:
            ep_entropy = [bmap.mean_entropy()]
            ep_kl      = [bmap.kl_from_truth(base.grid)]

        step_logs = []

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "toggle"}
            a = int(action) if np.isscalar(action) else int(action[0])

            agent_pos = tuple(base.agent_pos)
            goal_pos  = tuple(base.goal_pos)
            door = base.grid.get(*base.door_pos)
            door_open   = door is not None and door.type == "door" and door.is_open
            door_locked = door is not None and door.type == "door" and door.is_locked
            carrying = None if base.carrying is None else base.carrying.type

            log_line = (
                f"pos={agent_pos} | goal={goal_pos} | "
                f"door_open={door_open} | door_locked={door_locked} | "
                f"carrying={carrying} | action={a}:{action_names.get(a)}"
            )
            if args.only_success:
                step_logs.append(log_line)
            else:
                print(log_line)

            obs, reward, dones, infos = env.step(action)
            total += float(reward[0])
            done = bool(dones[0])
            step += 1

            if bmap is not None:
                ep_entropy.append(bmap.mean_entropy())
                ep_kl.append(bmap.kl_from_truth(base.grid))

        ep_attempted += 1
        is_success = infos[0].get("is_success", False)

        if args.only_success and not is_success:
            print(f"  [attempt {ep_attempted}] failed ({step} steps) — skipping")
            continue

        if args.only_success:
            for line in step_logs:
                print(line)

        ep_shown += 1
        print(f"Episode {ep_shown} finished in {step} steps")
        final_pos = tuple(infos[0].get("agent_pos"))
        goal_pos  = tuple(infos[0].get("goal_pos"))
        print(f"FINAL POS (from info): {final_pos} | GOAL POS: {goal_pos}")
        print("SUCCESS:", is_success)

        ep_rewards.append(total)
        if bmap is not None:
            all_entropy_histories.append(ep_entropy)
            all_kl_histories.append(ep_kl)

        if is_success:
            successes += 1

        if optimal_len > 0:
            ratio = optimal_len / step
            ep_optimality.append(ratio)
            print(f"PATH OPTIMALITY: {ratio:.3f}  (optimal={optimal_len}, actual={step})")
        else:
            print(f"PATH OPTIMALITY: N/A  (no valid path found)")

    env.close()

    if bmap is not None and all_entropy_histories:
        _plot_belief_metrics(all_entropy_histories, all_kl_histories, args.noise, args.model)

    if args.video:
        print(f"Videos saved to: {args.video_dir}")

    print(f"Episodes shown: {ep_shown}")
    if args.only_success:
        print(f"Success rate: {successes}/{ep_attempted} attempts = {successes/ep_attempted:.2%}")
    else:
        print(f"Success rate: {successes / ep_attempted:.2%}")
    print(f"Mean reward: {np.mean(ep_rewards):.3f} | Std: {np.std(ep_rewards):.3f}")
    if ep_optimality:
        print(f"Mean path optimality: {np.mean(ep_optimality):.3f} | Std: {np.std(ep_optimality):.3f}")


if __name__ == "__main__":
    main()
