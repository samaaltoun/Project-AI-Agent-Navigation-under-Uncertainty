import argparse
import json
import os
import shutil
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
from stable_baselines3.common.vec_env import DummyVecEnv

#from envs.simple_goal_env import register_env
from envs.simple_keydoor_env import register_env
from envs.wrappers import BeliefMapObsWrapper, ReducedActionSpace, SensorNoiseWrapper
from gymnasium.wrappers import FlattenObservation

class MiniGridCNN(BaseFeaturesExtractor):
  
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),#7x7 go 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),     #7x7 go 4x4
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),     #4x4 go 2x2
            nn.ReLU(),
            nn.Flatten(),)

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CSVLoggerCallback(BaseCallback):
    
    #Logs episode metrics to CSV and prints a rolling summary to console every log interval    
    ##Rolling window tracks (last 100 episodes):
        #success rate           
        #key rate       
        #door rate     
        #mean ep length 


    WINDOW = 100
    def __init__(self, log_path: Path, log_interval: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_interval = log_interval
        self.rows = []
        self._last_dump = time.time()
        self._last_print_step = 0
        self._success:     list = []
        self._picked_key:  list = []
        self._opened_door: list = []
        self._ep_len:      list = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue
            ep      = info["episode"]
            success = bool(info.get("is_success",   False))
            key     = bool(info.get("picked_key",   False))
            door    = bool(info.get("opened_door",  False))

            #update rolling buffers
            for buf, val in [
                (self._success,     float(success)),
                (self._picked_key,  float(key)),
                (self._opened_door, float(door)),
                (self._ep_len,      ep["l"]),
            ]:
                buf.append(val)
                if len(buf) > self.WINDOW:
                    buf.pop(0)

            self.rows.append({
                "timesteps":   self.num_timesteps,
                "ep_len":      ep["l"],
                "ep_rew":      ep["r"],
                "success":     int(success),
                "picked_key":  int(key),
                "opened_door": int(door),
            })

        #print summary 
        if (self.num_timesteps - self._last_print_step >= self.log_interval
                and self._success):
            self._last_print_step = self.num_timesteps
            n = len(self._success)
            print(
                f"\n[{self.num_timesteps:>9,} steps | last {n} eps]  "
                f"success={np.mean(self._success):.1%}  "
                f"key={np.mean(self._picked_key):.1%}  "
                f"door={np.mean(self._opened_door):.1%}  "
                f"ep_len={np.mean(self._ep_len):.0f}"
            )

        #clearing CSV every 2 secs 
        if time.time() - self._last_dump > 2.0 and self.rows:
            self._dump()
            self._last_dump = time.time()

        return True

    def _dump(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        header = "timesteps,ep_len,ep_rew,success,picked_key,opened_door\n"
        file_exists = self.log_path.exists()
        with self.log_path.open("a", encoding="utf-8") as f:
            if not file_exists:
                f.write(header)
            for r in self.rows:
                f.write(
                    f'{r["timesteps"]},{r["ep_len"]},{r["ep_rew"]},'
                    f'{r["success"]},{r["picked_key"]},{r["opened_door"]}\n'
                )
        self.rows.clear()

    def _on_training_end(self) -> None:
        if self.rows:
            self._dump()
        if self._success:
            n = len(self._success)
            print(
                f"\n===== TRAINING COMPLETE =====\n"
                f"Final {n}-episode window:\n"
                f"  Success rate : {np.mean(self._success):.1%}\n"
                f"  Key pickup   : {np.mean(self._picked_key):.1%}\n"
                f"  Door opened  : {np.mean(self._opened_door):.1%}\n"
                f"  Mean ep len  : {np.mean(self._ep_len):.0f}\n"
                f"============================="
            )


def make_env(noise_prob: float = 0.0, view_size: int = 5, grid_size: int = 15, num_walls: int = 0):
    from envs.simple_keydoor_env import SensorConfig
    sensor_cfg = SensorConfig(noise_prob=noise_prob)  # noise applied inside belief update
    env = gym.make(
        "SimpleKeyDoor-v0",
        render_mode=None,
        size=grid_size,
        num_extra_walls=num_walls,
        agent_view_size=view_size,
        sensor_cfg=sensor_cfg,        # pass noise directly into the environment
    )
    env = ReducedActionSpace(env)
    env = BeliefMapObsWrapper(env)   # agent observes the full belief map, not raw pixels
    env = FlattenObservation(env)    #flatten (9,H,W) to (9*H*W,) for MlpPolicy
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default=None,
                        help="Path to a level JSON file (from level_generator.py).")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to existing model to continue training from (noise curriculum).")
    parser.add_argument("--noise_prob", type=float, default=None,
                        help="Override noise_prob for this training run (e.g. 0.3 or 0.6).")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total training timesteps (e.g. 3000000). Default: 10000000.")
    args = parser.parse_args()

    register_env()

    # Load level config if provided
    level_cfg = {}
    if args.level:
        with open(args.level, "r", encoding="utf-8") as f:
            level_cfg = json.load(f)
        print(f"Loaded level config: {args.level}")

    # Save config
    config = {
        "env_id": "SimpleKeyDoor-v0",
        "algo": "PPO",
        "total_timesteps": args.timesteps if args.timesteps is not None else 10_000_000,
        "seed": 42,
        "policy": "MlpPolicy",   # flat belief map so  MLP is simpler and trains faster
        "n_steps": 4096,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "ent_coef": 0.02,          # low for Phase 3so confident decisions under heavy noise
        "n_epochs": 10,
        "noise_prob": args.noise_prob if args.noise_prob is not None else 0.0,
        "view_size": level_cfg.get("view_size", 5),
        "grid_size": level_cfg.get("grid_size", 13),
        "num_walls": level_cfg.get("num_walls", 0),
    }

    run_name = (
        time.strftime("%Y-%m-%d_%H%M%S")
        + f"_SimpleKeyDoor_PPO"
        + f"_grid{config['grid_size']}"
        + f"_view{config['view_size']}"
        + f"_walls{config['num_walls']}"
        + f"_noise{config['noise_prob']}"
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.level:
        shutil.copy(args.level, run_dir / "level.json")

    #FIXED: No fixed global seed so training envs see genuinely random layouts
    #np.random.seed(config["seed"])

    n_envs = 8
    #BeliefMapObsWrapper already outputs (C, H, W) eh VecTransposeImage not needed
    env = DummyVecEnv([
        lambda: make_env(
            noise_prob=config["noise_prob"],
            view_size=config["view_size"],
            grid_size=config["grid_size"],
            num_walls=config["num_walls"],
        )
        for _ in range(n_envs)
    ])

    #curriculum phase  using MlpPolicy, no CNN extractor needed ---
    # When switching back to CnnPolicy, restore policy_kwargs:
    # policy_kwargs = dict(
    #     features_extractor_class=MiniGridCNN,
    #     features_extractor_kwargs=dict(features_dim=128),
    # Larger network — 256x256 instead of default 64x64
    # needed for 1521-feature input (9×13×13)
    policy_kwargs = dict(net_arch=[256, 256])

    if args.finetune:
        print(f"Fine-tuning from: {args.finetune}")
        model = PPO.load(args.finetune, env=env)
        model.ent_coef = config["ent_coef"]
        model.learning_rate = config["learning_rate"]
    else:
        model = PPO(
            config["policy"],
            env,
            verbose=1,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            gae_lambda=0.95,
            ent_coef=config["ent_coef"],
            clip_range=0.2,
            n_epochs=config["n_epochs"],
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
        )

    csv_cb = CSVLoggerCallback(run_dir / "metrics.csv", log_interval=10_000)
    model.learn(total_timesteps=config["total_timesteps"], callback=csv_cb)

    model.save(str(run_dir / "model.zip"))
    print(f"Saved model to: {run_dir / 'model.zip'}")
    print(f"Saved metrics to: {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
