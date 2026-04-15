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
from functools import partial
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.simple_keydoor_env import register_env
from minigrid.wrappers import ViewSizeWrapper
from stable_baselines3.common.vec_env import VecTransposeImage
from envs.wrappers import ReducedActionSpace, SensorNoiseWrapper, BeliefObsWrapper
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


def make_env(
    noise_prob: float = 0.0,
    view_size: int = 7,
    grid_size: int = 13,
    n_lava: int = 0,
    tpr: float | None = None,
    decay: float = 0.0,
):
    register_env()   # each subprocess needs to register the env independently
    env = gym.make("SimpleKeyDoor-v0", render_mode=None, size=grid_size, n_lava=n_lava)
    env = ReducedActionSpace(env)
    if noise_prob > 0.0:
        env = SensorNoiseWrapper(env, noise_prob=noise_prob)
    if view_size != 7:
        env = ViewSizeWrapper(env, agent_view_size=view_size)
    # BeliefObsWrapper replaces ImgObsWrapper — it maintains the belief map
    # internally and outputs the belief tensor as the observation.
    env = BeliefObsWrapper(
        env,
        grid_size=grid_size,
        noise_prob=noise_prob,
        tpr=tpr,
        decay=decay,
        view_size=view_size,
    )
    env = Monitor(env)
    return env


def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    """
    Returns a callable that computes the learning rate at each step.
    Linearly decays from initial_lr to final_lr over the course of training.
    progress_remaining goes from 1.0 (start) to 0.0 (end).
    """
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


def main():
    register_env()

    # Save config
    config = {
        "env_id": "SimpleKeyDoor-v0",
        "algo": "PPO",
        "total_timesteps": 3_000_000,
        "seed": 42,
        "policy": "CnnPolicy",  # MiniGrid obs is image-like -> use CNN
        "n_steps": 1024,          # number of steps to run in each environment per update (i.e. batch size = n_steps * n_envs)
        "batch_size": 512,          # number of samples per gradient update (max 512 here since we have n_envs=1)
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "ent_coef": 0.05,
        "n_epochs": 4,
        "noise_prob": 0.0,      # sensor noise level [0.0, 1.0]
        "view_size": 7,         # agent field-of-view (default 7x7)
        "grid_size": 8,         # environment grid size (small grid for faster learning)
        "n_lava": 0,            # lava tiles scattered across the grid
        "tpr": None,            # true positive rate (None = derive from noise_prob)
        "decay": 0.01,          # belief decay per step (0.0 = perfect memory)
        "finetune_from": None,  # set to None to train from scratch
    }

    run_name = time.strftime("%Y-%m-%d_%H%M%S") + f"_SimpleKeyDoor_PPO_noise{config['noise_prob']}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    np.random.seed(config["seed"])

    n_envs = 8
    env_fn = partial(
        make_env,
        noise_prob=config["noise_prob"],
        view_size=config["view_size"],
        grid_size=config["grid_size"],
        n_lava=config["n_lava"],
        tpr=config["tpr"],
        decay=config["decay"],
    )
    env = SubprocVecEnv([env_fn for _ in range(n_envs)])
    # BeliefObsWrapper already outputs channels-first (C,H,W), so we
    # skip VecTransposeImage (which rejects observations with >4 channels).

    policy_kwargs = dict(
    features_extractor_class=MiniGridCNN,
    features_extractor_kwargs=dict(features_dim=128),
    )

    if config.get("finetune_from"):
        print(f"Fine-tuning from: {config['finetune_from']}")
        model = PPO.load(
            config["finetune_from"],
            env=env,
            custom_objects={"features_extractor_class": MiniGridCNN},
            verbose=1,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            ent_coef=config["ent_coef"],
            n_epochs=config["n_epochs"],
        )
    else:
        model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        learning_rate=linear_schedule(config["learning_rate"]),
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
