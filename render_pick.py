#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
import logging
import genesis as gs
import igl
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces
from stable_baselines3 import SAC

# ─── libigl monkey-patch ───────────────────────────────────────────────────────
_original_sd = igl.signed_distance
def _patched_sd(*args, **kwargs):
    r = _original_sd(*args, **kwargs)
    if isinstance(r, tuple) and len(r) >= 3:
        return r[0], r[1], r[2]
    return r
igl.signed_distance = _patched_sd
# ──────────────────────────────────────────────────────────────────────────────

class FrankaGraspRenderEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, num_objects=1):
        super().__init__()
        # Genesis 初期化
        try:
            gs.init(backend=gs.cuda)
        except Exception as e:
            if "already initialized" not in str(e):
                raise
        logging.getLogger("genesis").setLevel(logging.WARNING)

        # シーン構築（レンダー有効）
        self.scene = gs.Scene(show_viewer=True)
        self.scene.add_entity(gs.morphs.Plane())
        self.scene.add_entity(gs.morphs.Box(size=[0.8,0.8,0.05], pos=[0.5,0.0,0.025]))
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

        # 把持対象キューブ群を配置
        self.objects = []
        for _ in range(num_objects):
            x = np.random.uniform(0.3,0.7)
            y = np.random.uniform(-0.3,0.3)
            z = 0.05 + 0.02
            cube = self.scene.add_entity(
                gs.morphs.Box(size=[0.04,0.04,0.04], pos=[x,y,z])
            )
            self.objects.append(cube)

        # ビルド
        self.scene.build()

        # 観測／行動空間
        full_q = self.robot.get_dofs_position().cpu().numpy()
        obs_dim = len(full_q) + 3 * len(self.objects)
        act_dim = 7
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32)

        # 制御する関節インデックス（末尾7軸）
        total = len(full_q)
        self.joint_idx = list(range(total-act_dim, total))

    def reset(self, *, seed=None, options=None):
        self.scene.reset()
        return self._get_obs(), {}

    def step(self, action):
        self.robot.control_dofs_velocity(action.tolist(), self.joint_idx)
        self.scene.step()
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, False, {}

    def _get_obs(self):
        q = self.robot.get_dofs_position().cpu().numpy()
        pos = [o.get_dofs_position().cpu().numpy()[:3] for o in self.objects]
        return np.concatenate([q, *pos]).astype(np.float32)

    def _compute_reward(self):
        eef = self.robot.get_dofs_position().cpu().numpy()[-3:]
        for o in self.objects:
            p = o.get_dofs_position().cpu().numpy()[:3]
            if np.linalg.norm(p - eef) < 0.06 and p[2] > 0.1:
                return 1.0, True
        return 0.0, False

if __name__ == "__main__":
    # 学習済みモデルのパス
    MODEL_PATH = "/home/knishizawa/genesis_sim/models/grasp_ep0250.zip"

    # 環境生成
    fps = FrankaGraspRenderEnv.metadata["render_fps"]
    env = FrankaGraspRenderEnv(num_objects=2)
    env = TimeLimit(env, max_episode_steps=fps * 15)

    # モデルロード
    model = SAC.load(MODEL_PATH, device="cuda")

    # 実行
    obs, _ = env.reset()
    start = time.time()
    while time.time() - start < 10.0:  # 10秒間レンダリング
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        if done:
            print("Grasp success!")
            break

    try:
        env.scene.close()
    except:
        pass
