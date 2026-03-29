# train_ppo_curriculum.py
# Two-stage curriculum training:
#   Stage 1 — 1 wall, learn basic key/door/goal chain
#   Stage 2 — 2 walls, load stage 1 model and continue
# the purpose of this file is to see whether training the simpler 1-wall version first can help the agent 
# learn the harder 2-wall version faster than training from scratch as previously training for the
# 2-wall version from scratch took a long time and had a 0 success rate

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

from envs.variational_goal_env import LevelConfig, VariationalGoalEnv, register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper
from minigrid.wrappers import ImgObsWrapper, ViewSizeWrapper


# -----------------------------------------------------------------------
# CNN
# -----------------------------------------------------------------------

class MiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# -----------------------------------------------------------------------
# CSV Logger
# -----------------------------------------------------------------------

class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_path: Path, stage: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.stage = stage
        self.rows = []
        self._last_dump = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.rows.append({
                    "stage": self.stage,
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
                f.write("stage,timesteps,ep_len,ep_rew\n")
            for r in self.rows:
                f.write(f'{r["stage"]},{r["timesteps"]},{r["ep_len"]},{r["ep_rew"]}\n')
        self.rows.clear()

    def _on_training_end(self):
        if self.rows:
            self._dump()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

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


def make_env(level: LevelConfig, noise_prob: float = 0.0, view_size: int = 11):
    env = VariationalGoalEnv(level=level, render_mode=None)
    env = ReducedActionSpace(env)
    if noise_prob > 0.0:
        env = SensorNoiseWrapper(env, noise_prob=noise_prob)
    if view_size != 7:
        env = ViewSizeWrapper(env, agent_view_size=view_size)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env


def build_vec_env(level: LevelConfig, n_envs: int, noise_prob: float, view_size: int):
    env = DummyVecEnv([
        lambda l=level: make_env(level=l, noise_prob=noise_prob, view_size=view_size)
        for _ in range(n_envs)
    ])
    return VecTransposeImage(env)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    # ----------------------------------------------------------------
    # Shared config
    # ----------------------------------------------------------------
    config = {
        "algo": "PPO",
        "seed": 42,
        "policy": "CnnPolicy",
        "n_steps": 1024,
        "batch_size": 512,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "ent_coef": 0.2,
        "n_epochs": 10,
        "noise_prob": 0.0,
        "view_size": 11,
        "n_envs": 8,
    }

    # ----------------------------------------------------------------
    # Stage configs
    # ----------------------------------------------------------------
    stage1_level = LevelConfig(
        width=20,
        height=16,
        num_walls=1,              # simpler — one key, one door
        doors_per_wall=1,
        wall_orientation="vertical",
        min_gap=3,
    )
    stage1_timesteps = 2_000_000

    stage2_level = LevelConfig(
        width=20,
        height=16,
        num_walls=2,              # full task — two keys, two doors
        doors_per_wall=1,
        wall_orientation="vertical",
        min_gap=3,
    )
    stage2_timesteps = 3_000_000

    # ----------------------------------------------------------------
    # Run folder
    # ----------------------------------------------------------------
    register_env()

    env_name = "CurriculumGoal"
    runs_dir = Path("runs")
    date_str = time.strftime("%Y-%m-%d")
    time_str = time.strftime("%Y-%m-%d_%H%M%S")
    run_id = _next_run_id(runs_dir, date_str, env_name)
    run_dir = runs_dir / f"{time_str}_{env_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        **config,
        "stage1": {
            "timesteps": stage1_timesteps,
            "level": {
                "width": stage1_level.width,
                "height": stage1_level.height,
                "num_walls": stage1_level.num_walls,
                "doors_per_wall": stage1_level.doors_per_wall,
                "wall_orientation": stage1_level.wall_orientation,
                "min_gap": stage1_level.min_gap,
            },
        },
        "stage2": {
            "timesteps": stage2_timesteps,
            "level": {
                "width": stage2_level.width,
                "height": stage2_level.height,
                "num_walls": stage2_level.num_walls,
                "doors_per_wall": stage2_level.doors_per_wall,
                "wall_orientation": stage2_level.wall_orientation,
                "min_gap": stage2_level.min_gap,
            },
        },
    }
    (run_dir / "config.json").write_text(json.dumps(full_config, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}")

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])

    policy_kwargs = dict(
        features_extractor_class=MiniGridCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # ================================================================
    # STAGE 1 — 1 wall
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 1 — 1 wall  |  target: learn key → door → goal")
    print("=" * 60)

    env1 = build_vec_env(stage1_level, config["n_envs"], config["noise_prob"], config["view_size"])

    model = PPO(
        "CnnPolicy",
        env1,
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

    cb1 = CSVLoggerCallback(run_dir / "metrics.csv", stage=1)
    model.learn(total_timesteps=stage1_timesteps, callback=cb1)

    stage1_path = run_dir / "model_stage1.zip"
    model.save(str(stage1_path))
    env1.close()
    print(f"Stage 1 complete — saved to {stage1_path}")

    # ================================================================
    # STAGE 2 — 2 walls (load stage 1 weights)
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 2 — 2 walls  |  loading stage 1 weights")
    print("=" * 60)

    env2 = build_vec_env(stage2_level, config["n_envs"], config["noise_prob"], config["view_size"])

    model = PPO.load(str(stage1_path), env=env2)
    model.ent_coef = 0.15   # slightly reduce entropy for stage 2

    cb2 = CSVLoggerCallback(run_dir / "metrics.csv", stage=2)
    model.learn(total_timesteps=stage2_timesteps, callback=cb2, reset_num_timesteps=False)

    stage2_path = run_dir / "model_stage2.zip"
    model.save(str(stage2_path))
    env2.close()
    print(f"Stage 2 complete — saved to {stage2_path}")

    # ----------------------------------------------------------------
    print(f"\nAll done.")
    print(f"Stage 1 model → {stage1_path}")
    print(f"Stage 2 model → {stage2_path}")
    print(f"Metrics       → {run_dir / 'metrics.csv'}")
    print(f"Config        → {run_dir / 'config.json'}")


if __name__ == "__main__":
    main()