#!/usr/bin/env python3
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from tqdm import tqdm
import logging

# suppress Genesis INFO logs printed to stdout
_orig_write = sys.stdout.write
def _filtered_write(text):
    if "[Genesis]" in text:
        return
    _orig_write(text)
sys.stdout.write = _filtered_write

# Monkey-patch libigl
import igl
_original_signed_distance = igl.signed_distance
def _patched_signed_distance(*args, **kwargs):
    res = _original_signed_distance(*args, **kwargs)
    if isinstance(res, tuple) and len(res) >= 3:
        return res[0], res[1], res[2]
    return res
igl.signed_distance = _patched_signed_distance

import genesis as gs
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from dreamer.agents.dreamer_agent import DreamerAgent
import importlib

if importlib.util.find_spec("diffusion_policy"):
    from diffusion_policy import DiffusionPolicy
else:
    DiffusionPolicy = None

class FrankaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self, render=False):
        super().__init__()
        try:
            gs.init(backend=gs.cuda)
        except Exception as e:
            if "already initialized" not in str(e):
                raise
        logging.getLogger("genesis").setLevel(logging.WARNING)

        self.scene = gs.Scene(show_viewer=render)
        _ = self.scene.add_entity(gs.morphs.Plane())
        self.robot  = self.scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
        self.obj    = self.scene.add_entity(gs.morphs.Sphere(radius=0.02, pos=[0.5,0,0.02]))
        self.target = self.scene.add_entity(gs.morphs.Box(size=[0.05,0.05,0.001], pos=[0.6,0,0.001]))
        self.scene.build()

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

        q = self.robot.get_dofs_position()
        q = q.cpu().numpy() if isinstance(q, torch.Tensor) else np.array(q)
        total_dofs = len(q)
        joint_dim  = self.action_space.shape[0]
        self.joint_idx = list(range(total_dofs - joint_dim, total_dofs))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.scene.reset()
        return self._get_obs(), {}

    def step(self, action):
        self.robot.control_dofs_velocity(action.tolist(), self.joint_idx)
        self.scene.step()
        obs = self._get_obs()
        reward, success = self._compute_reward()
        return obs, reward, bool(success), False, {}

    def _get_obs(self):
        q = self.robot.get_dofs_position()
        q = q.cpu().numpy() if isinstance(q, torch.Tensor) else np.array(q)
        q = q[self.joint_idx].astype(np.float32)
        o = self.obj.get_dofs_position()
        o = o.cpu().numpy() if isinstance(o, torch.Tensor) else np.array(o)
        o = o[:3].astype(np.float32)
        return np.concatenate([q, o])

    def _compute_reward(self):
        o = self.obj.get_dofs_position()
        o = o.cpu().numpy() if isinstance(o, torch.Tensor) else np.array(o)
        o = o[:3]
        t = self.target.get_dofs_position()
        t = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
        t = t[:3]
        success = float(np.linalg.norm(o - t) < 0.05)
        return success, success == 1.0

def make_env(render=False):
    def _init():
        return FrankaPickPlaceEnv(render=render)
    return _init

def train_and_evaluate(constructor, n_envs=4, total_timesteps=1_000_000):
    # ４並列、モニタ付き
    vecenv = SubprocVecEnv([make_env(render=False) for _ in range(n_envs)])
    vecenv = VecMonitor(vecenv)

    # SAC on GPU
    model = constructor(vecenv)
    # デバイスは PyTorch default (cuda) になります
    model.learn(total_timesteps=total_timesteps)
    return model

def main():
    # 並列数はお好みで
    n_envs = 4
    total_timesteps = 1_000_000

    constructor = lambda e: SAC(
        'MlpPolicy', e,
        verbose=1,
        device='cuda',            # ネットワーク更新を GPU で
        tensorboard_log='./tb/'    # TensorBoard ログ
    )
    print(f"--- Training SAC with {n_envs} parallel envs ---")
    model = train_and_evaluate(constructor, n_envs=n_envs, total_timesteps=total_timesteps)

    # 終了後は評価
    eval_env = FrankaPickPlaceEnv(render=True)
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = eval_env.step(action)
    print("Evaluation done.")

if __name__ == "__main__":
    main()
