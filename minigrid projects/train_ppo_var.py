import os
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

from envs.variational_goal_env import VariationalGoalEnv, register_env
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper
from minigrid.wrappers import ImgObsWrapper, ViewSizeWrapper


class MiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
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


class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.rows = []
        self._last_dump = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
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
        header = "timesteps,ep_len,ep_rew\n"
        file_exists = self.log_path.exists()
        with self.log_path.open("a", encoding="utf-8") as f:
            if not file_exists:
                f.write(header)
            for r in self.rows:
                f.write(f'{r["timesteps"]},{r["ep_len"]},{r["ep_rew"]}\n')
        self.rows.clear()

    def _on_training_end(self) -> None:
        if self.rows:
            self._dump()


def _next_run_id(runs_dir: Path, date_str: str, env_name: str) -> str:
    """
    Scan the runs folder and returns the next run ID for today's date + env,
    """
    prefix = f"{date_str}_"
    existing = [
        d.name for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and env_name in d.name
    ] if runs_dir.exists() else []

    max_id = 0
    for name in existing:
        # name format: 2025-01-28_143022_VariationalGoal_run001
        parts = name.rsplit("run", 1)
        if len(parts) == 2 and parts[1].isdigit():
            max_id = max(max_id, int(parts[1]))

    return f"run{max_id + 1:03d}"


def make_env(grid_size: int = 10, noise_prob: float = 0.0, view_size: int = 7):
    env = gym.make("VariationalGoal-v0", render_mode=None)
    env = ReducedActionSpace(env)
    if noise_prob > 0.0:
        env = SensorNoiseWrapper(env, noise_prob=noise_prob)
    if view_size != 7:
        env = ViewSizeWrapper(env, agent_view_size=view_size)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env


def main():
    register_env()

    config = {
        "env_id": "VariationalGoal-v0",
        "algo": "PPO",
        "total_timesteps": 2_000_000,
        "seed": 42,
        "policy": "CnnPolicy",
        "n_steps": 512,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "ent_coef": 0.05,
        "n_epochs": 10,
        "noise_prob": 0.0,
        "view_size": 7,
        "grid_size": 10,
    }

    # --- Build run folder: runs/2025-01-28_143022_VariationalGoal_run001 ---
    env_name = "VariationalGoal"
    runs_dir = Path("runs")
    date_str = time.strftime("%Y-%m-%d")
    time_str = time.strftime("%Y-%m-%d_%H%M%S")
    run_id = _next_run_id(runs_dir, date_str, env_name)
    run_name = f"{time_str}_{env_name}_{run_id}"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}")

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])

    n_envs = 8
    env = DummyVecEnv([
        lambda: make_env(
            grid_size=config["grid_size"],
            noise_prob=config["noise_prob"],
            view_size=config["view_size"],
        )
        for _ in range(n_envs)
    ])
    env = VecTransposeImage(env)

    policy_kwargs = dict(
        features_extractor_class=MiniGridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

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
    print(f"Saved model    → {run_dir / 'model.zip'}")
    print(f"Saved metrics  → {run_dir / 'metrics.csv'}")
    print(f"Saved config   → {run_dir / 'config.json'}")


if __name__ == "__main__":
    main()
