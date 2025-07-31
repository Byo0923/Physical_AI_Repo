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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from tqdm import trange

# ─── libigl monkey-patch ───────────────────────────────────────────────────────
_original_sd = igl.signed_distance
def _patched_sd(*args, **kwargs):
    r = _original_sd(*args, **kwargs)
    if isinstance(r, tuple) and len(r) >= 3:
        return r[0], r[1], r[2]
    return r
igl.signed_distance = _patched_sd
# ──────────────────────────────────────────────────────────────────────────────

# 学習設定
NUM_ENVS          = 24       # 並列環境数
NUM_OBJECTS       = 1        # 球体の数
MAX_STEPS_EPISODE = 900      # エピソード長
TOTAL_EPISODES    = 500      # 総エピソード数
SAVE_DIR          = "models"
TB_LOGDIR         = "tensorboard_logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TB_LOGDIR, exist_ok=True)

def make_env(rank: int):
    def _init():
        try:
            gs.init(backend=gs.cuda)
        except Exception as e:
            if "already initialized" not in str(e):
                raise
        logging.getLogger("genesis").setLevel(logging.WARNING)
        scene = gs.Scene(show_viewer=False)
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=[0.8,0.8,0.05], pos=[0.5,0.0,0.025]))
        robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        objects = []
        for _ in range(NUM_OBJECTS):
            x, y = np.random.uniform(0.3,0.7), np.random.uniform(-0.3,0.3)
            radius = 0.03
            sphere = scene.add_entity(gs.morphs.Sphere(radius=radius, pos=[x,y,radius+0.05]))
            objects.append(sphere)
        scene.build()
        full_q = robot.get_dofs_position().cpu().numpy()
        obs_dim = len(full_q) + 3 * NUM_OBJECTS
        act_dim = 7

        class GraspSphereEnv(gym.Env):
            metadata = {"render_modes":["human"], "render_fps":60}
            def __init__(self):
                self.scene = scene
                self.robot = robot
                self.objects = objects
                self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
                self.action_space      = spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32)
                self.joint_idx = list(range(len(full_q)-act_dim, len(full_q)))

            def reset(self, *, seed=None, options=None):
                self.scene.reset()
                return self._get_obs(), {}

            def step(self, action):
                self.robot.control_dofs_velocity(action.tolist(), self.joint_idx)
                self.scene.step()
                obs = self._get_obs()
                eef = self.robot.get_dofs_position().cpu().numpy()[-3:]
                reward, done = 0.0, False
                for o in self.objects:
                    p = o.get_dofs_position().cpu().numpy()[:3]
                    dist = np.linalg.norm(p - eef)
                    r = 1.0 - np.tanh(10 * dist)
                    reward = max(reward, r)
                if dist < 0.05:
                    done = True
                return obs, reward, done, False, {}

            def _get_obs(self):
                q = self.robot.get_dofs_position().cpu().numpy()
                pos = [o.get_dofs_position().cpu().numpy()[:3] for o in self.objects]
                return np.concatenate([q, *pos]).astype(np.float32)

        env = GraspSphereEnv()
        return TimeLimit(env, max_episode_steps=MAX_STEPS_EPISODE)
    return _init

if __name__ == "__main__":
    # 訓練／評価環境の準備
    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    train_env = VecMonitor(train_env, SAVE_DIR)
    eval_env  = DummyVecEnv([make_env(0)])

    # ロガー＆モデル
    new_logger = configure(TB_LOGDIR, ["stdout", "tensorboard"] )
    model = SAC(
        policy="MlpPolicy", env=train_env, verbose=1,
        device="cuda", tensorboard_log=TB_LOGDIR,
        batch_size=256, learning_rate=3e-4, gamma=0.99
    )
    model.set_logger(new_logger)

    # コールバック設定
    checkpoint_cb = CheckpointCallback(
        save_freq=MAX_STEPS_EPISODE * NUM_ENVS * 10,
        save_path=SAVE_DIR, name_prefix="sac_grasp_sphere"
    )
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=SAVE_DIR,
        log_path=TB_LOGDIR, eval_freq=MAX_STEPS_EPISODE * NUM_ENVS,
        deterministic=True
    )

    # 初期モデル保存
    model.save(f"{SAVE_DIR}/grasp_sphere_init")

    # エピソード単位で学習進行表示
    for ep in trange(1, TOTAL_EPISODES+1, desc="Episode"):
        model.learn(
            total_timesteps=MAX_STEPS_EPISODE * NUM_ENVS,
            reset_num_timesteps=False,
            callback=[checkpoint_cb, eval_cb]
        )
        # 進捗表示には内部の tqdm を利用可能

                # 定期保存
        if ep % 10 == 0:
            model.save(f"{SAVE_DIR}/grasp_sphere{ep:04d}")
    # 最終モデル保存
    model.save(f"{SAVE_DIR}/grasp_sphere_final")
    print("Training complete.")
