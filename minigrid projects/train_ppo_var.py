# this file will train a PPO agent on the VariationalGoalEnv and save the model, config, and training metrics to a timestamped folder in `runs`
# It generates a new grid layout on every episode, so the agent must learn to generalize across different maps with the same underlying structure

# train_ppo_var.py
import json
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from envs.variational_goal_env import LevelConfig, VariationalGoalEnv, register_env, load_snapshot
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper
from minigrid.wrappers import ImgObsWrapper, ViewSizeWrapper


class MiniGridCNN(BaseFeaturesExtractor):
    """
    MiniGrid-optimised CNN using 2x2 kernels with stride 1 (no padding).
    Preserves spatial information much better than stride-2 on small 7x7 inputs.
    Source: https://minigrid.farama.org/content/training/
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),  # 7x7 → 6x6
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),                # 6x6 → 5x5
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),                # 5x5 → 4x4
            nn.ReLU(),
            nn.Flatten(),                              # 64 * 4 * 4 = 1024
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.rows = []
        self._last_dump = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.rows.append({
                    "timesteps": self.num_timesteps,
                    "ep_len": ep["l"],
                    "ep_rew": ep["r"],
                })
        if time.time() - self._last_dump > 2.0 and self.rows:
            self._dump()
            self._last_dump = time.time()
        return True

    def _dump(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.log_path.exists()
        with self.log_path.open("a", encoding="utf-8") as f:
            if not file_exists:
                f.write("timesteps,ep_len,ep_rew\n")
            for r in self.rows:
                f.write(f'{r["timesteps"]},{r["ep_len"]},{r["ep_rew"]}\n')
        self.rows.clear()

    def _on_training_end(self):
        if self.rows:
            self._dump()


def _next_run_id(runs_dir: Path, date_str: str, env_name: str) -> str:
    prefix = f"{date_str}_"
    existing = [
        d.name for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and env_name in d.name
    ] if runs_dir.exists() else []
    max_id = 0
    for name in existing:
        parts = name.rsplit("run", 1)
        if len(parts) == 2 and parts[1].isdigit():
            max_id = max(max_id, int(parts[1]))
    return f"run{max_id + 1:03d}"


def make_env(level: LevelConfig, noise_prob: float = 0.0, view_size: int = 11, snapshot=None):
    env = VariationalGoalEnv(snapshot=snapshot, render_mode=None)
    env = ReducedActionSpace(env)
    if noise_prob > 0.0:
        env = SensorNoiseWrapper(env, noise_prob=noise_prob)
    if view_size != 7:
        env = ViewSizeWrapper(env, agent_view_size=view_size)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env


def main():
    # ----------------------------------------------------------------
    # Level layout — edit these to change the map
    # ----------------------------------------------------------------
    level = LevelConfig(
        width=20,               # grid width  (including border walls)
        height=16,              # grid height (including border walls)
        num_walls=2,            # number of dividing walls (= number of room dividers)
        doors_per_wall=1,       # locked doors (and matching keys) per wall
        wall_orientation="vertical",  # 'vertical' | 'horizontal' | 'random'
        min_gap=3,              # minimum room width/height in cells
    )

    # ----------------------------------------------------------------
    # Training hyperparameters
    # ----------------------------------------------------------------
    config = {
        "env_id": "VariationalGoal-v0",
        "algo": "PPO",
        "total_timesteps": 2_000_000,
        "seed": 42,
        "policy": "CnnPolicy",
        "n_steps": 128,
        "batch_size": 64,
        "learning_rate": 9e-4,
        "gamma": 0.99,
        "ent_coef": 0.1,
        "n_epochs": 10,
        "noise_prob": 0.0,
        "view_size": 7,
        # Level config saved alongside for reference
        "level": {
            "width": level.width,
            "height": level.height,
            "num_walls": level.num_walls,
            "doors_per_wall": level.doors_per_wall,
            "wall_orientation": level.wall_orientation,
            "min_gap": level.min_gap,
        },
    }

    register_env()

    # ----------------------------------------------------------------
    # Run folder:  runs/2025-01-28_143022_VariationalGoal_run001
    # ----------------------------------------------------------------
    env_name = "VariationalGoal"
    runs_dir = Path("runs")
    date_str = time.strftime("%Y-%m-%d")
    time_str = time.strftime("%Y-%m-%d_%H%M%S")
    run_id = _next_run_id(runs_dir, date_str, env_name)
    run_dir = runs_dir / f"{time_str}_{env_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}")

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])

    n_envs = 8
    snap = load_snapshot(Path(__file__).parent / "levels/stage2small.json")
    env = DummyVecEnv([
        lambda: make_env(level=level, noise_prob=config["noise_prob"], view_size=config["view_size"], snapshot=snap)
        for _ in range(n_envs)
    ])
    env = VecTransposeImage(env)

    policy_kwargs = dict(
        features_extractor_class=MiniGridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # ----------------------------------------------------------------
    # Optionally warm-start from a pretrained model (curriculum).
    # Set pretrained_model to the path of a stage-1 model.zip to
    # transfer its weights instead of training from scratch.
    # ----------------------------------------------------------------
    pretrained_model = "runs/2026-03-29_222036_VariationalGoal_run009/model.zip"  # set to a model path for curriculum warm-start

    if pretrained_model is not None:
        pretrained_model = str(pretrained_model).removesuffix(".zip")
        print(f"Loading pretrained weights from: {pretrained_model}")
        model = PPO.load(pretrained_model, env=env)
        model.ent_coef = config["ent_coef"]  # reset entropy for new task
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            seed=config["seed"],
            gae_lambda=0.95,
            ent_coef=config["ent_coef"],
            clip_range=0.2,
            n_epochs=config["n_epochs"],
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
        )

    csv_cb = CSVLoggerCallback(run_dir / "metrics.csv")
    model.learn(total_timesteps=config["total_timesteps"], callback=csv_cb)

    model.save(str(run_dir / "model.zip"))
    print(f"Saved model   → {run_dir / 'model.zip'}")
    print(f"Saved metrics → {run_dir / 'metrics.csv'}")
    print(f"Saved config  → {run_dir / 'config.json'}")


if __name__ == "__main__":
    main()
