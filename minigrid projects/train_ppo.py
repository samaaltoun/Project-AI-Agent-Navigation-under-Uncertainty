import os
import json
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

#from envs.simple_goal_env import register_env
from envs.simple_keydoor_env import register_env
from minigrid.wrappers import ImgObsWrapper, ViewSizeWrapper
from stable_baselines3.common.vec_env import VecTransposeImage
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper
from envs.extractors import MiniGridCNN


class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.rows = []
        self._last_dump = time.time()

    def _on_step(self) -> bool:
        # SB3 stores episode info in infos when using Monitor
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                self.rows.append({
                    "timesteps": self.num_timesteps,
                    "ep_len": ep["l"],
                    "ep_rew": ep["r"],
                })

        # dump every ~2 seconds (keeps file updated)
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


def make_env(noise_prob: float = 0.0, view_size: int = 7):
    env = gym.make("SimpleKeyDoor-v0", render_mode=None)
    env = ReducedActionSpace(env)
    if noise_prob > 0.0:
        env = SensorNoiseWrapper(env, noise_prob=noise_prob)
    if view_size != 7:
        env = ViewSizeWrapper(env, agent_view_size=view_size)
    env = ImgObsWrapper(env)    # strips dict → keeps only image array; must be last obs wrapper
    env = Monitor(env)
    return env


def main():
    register_env()

    # Save config
    config = {
        "env_id": "SimpleKeyDoor-v0",
        "algo": "PPO",
        "total_timesteps": 300_000,
        "seed": 42,
        "policy": "CnnPolicy",  # MiniGrid obs is image-like -> use CNN
        "n_steps": 512,          # number of steps to run in each environment per update (i.e. batch size = n_steps * n_envs)
        "batch_size": 256,          # number of samples per gradient update (max 512 here since we have n_envs=1)
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "ent_coef": 0.05,
        "n_epochs": 10,
        "noise_prob": 0.2,      # sensor noise level [0.0, 1.0]
        "view_size": 7,         # agent field-of-view (default 7x7)
    }

    run_name = time.strftime("%Y-%m-%d_%H%M%S") + f"_SimpleKeyDoor_PPO_noise{config['noise_prob']}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    np.random.seed(config["seed"])

    n_envs = 8
    env = DummyVecEnv([lambda: make_env(noise_prob=config["noise_prob"], view_size=config["view_size"]) for _ in range(n_envs)])
    env = VecTransposeImage(env)  # <-- converts (H,W,C) to (C,H,W) for CnnPolicy

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
    print(f"Saved model to: {run_dir / 'model.zip'}")
    print(f"Saved metrics to: {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()