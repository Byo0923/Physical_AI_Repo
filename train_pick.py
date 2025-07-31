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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm import trange

# ─── libigl monkey‑patch ───────────────────────────────────────────────────────
_original_sd = igl.signed_distance
def _patched_sd(*args, **kwargs):
    r = _original_sd(*args, **kwargs)
    if isinstance(r, tuple) and len(r) >= 3:
        return r[0], r[1], r[2]
    return r
igl.signed_distance = _patched_sd
# ──────────────────────────────────────────────────────────────────────────────

# 学習設定
NUM_ENVS          = 24       # 並列数
NUM_OBJECTS       = 2       # テーブル上のキューブ数
MAX_STEPS_EPISODE = 900     # 1エピソードあたりのステップ数
TOTAL_EPISODES    = 500     # エピソード数
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
            x = np.random.uniform(0.3,0.7)
            y = np.random.uniform(-0.3,0.3)
            z = 0.05 + 0.02
            cube = scene.add_entity(gs.morphs.Box(size=[0.04,0.04,0.04], pos=[x,y,z]))
            objects.append(cube)
        scene.build()
        full_q = robot.get_dofs_position().cpu().numpy()
        obs_dim = len(full_q) + 3 * NUM_OBJECTS
        act_dim = 7

        class GraspEnv(gym.Env):
            metadata = {"render_modes":["human"], "render_fps":60}
            def __init__(self):
                self.scene = scene
                self.robot = robot
                self.objects = objects
                self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
                self.action_space      = spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32)
                total = len(full_q)
                self.joint_idx = list(range(total-act_dim, total))
            def reset(self, *, seed=None, options=None):
                self.scene.reset()
                return self._get_obs(), {}
            def step(self, action):
                self.robot.control_dofs_velocity(action.tolist(), self.joint_idx)
                self.scene.step()
                obs = self._get_obs()
                # 距離で把持成功判定
                eef = self.robot.get_dofs_position().cpu().numpy()[-3:]
                reward = 0.0; done=False
                for o in self.objects:
                    p = o.get_dofs_position().cpu().numpy()[:3]
                    if np.linalg.norm(p-eef)<0.06 and p[2]>0.1:
                        reward=1.0; done=True; break
                return obs, reward, done, False, {}
            def _get_obs(self):
                q = self.robot.get_dofs_position().cpu().numpy()
                pos = [o.get_dofs_position().cpu().numpy()[:3] for o in self.objects]
                return np.concatenate([q, *pos]).astype(np.float32)

        env = GraspEnv()
        env = TimeLimit(env, max_episode_steps=MAX_STEPS_EPISODE)
        return env
    return _init

if __name__ == "__main__":
    # 並列環境
    env_fns = [make_env(i) for i in range(NUM_ENVS)]
    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env, SAVE_DIR)

    # TensorBoard logger
    new_logger = configure(TB_LOGDIR, ["stdout", "tensorboard"])

    # SAC エージェントの作成
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        device="cuda",
        tensorboard_log=TB_LOGDIR,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99
    )
    model.set_logger(new_logger)

    # チェックポイント
    checkpoint_cb = CheckpointCallback(
        save_freq=MAX_STEPS_EPISODE * NUM_ENVS * 10,  # 10エピソードごと
        save_path=SAVE_DIR,
        name_prefix="sac_grasp"
    )

    # 学習ループ
    cumulative_virtual_steps = 0
    start_time = time.time()
    model_save_ep = 0
    model.save(f"{SAVE_DIR}/grasp_ep{model_save_ep:04d}")

    for ep in trange(1, TOTAL_EPISODES+1, desc="Ep"):
        # この learn でちょうど1エピソード分収集
        model.learn(
            total_timesteps=MAX_STEPS_EPISODE,
            reset_num_timesteps=False,
            callback=checkpoint_cb
        )
        cumulative_virtual_steps += MAX_STEPS_EPISODE * NUM_ENVS
        real_elapsed = time.time() - start_time

        # 評価：ランダム1エピソードで成功判定
        ep_success = 0
        obs = train_env.reset()                   # obs.shape == (NUM_ENVS, obs_dim)
        for _ in range(MAX_STEPS_EPISODE):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = train_env.step(action)
            # you’ll get arrays; e.g. rewards is shape (NUM_ENVS,)
            # pick one env to evaluate, or compute success rate across them
            if np.any((rewards > 0) & dones):
                ep_success = 1
                break
        
        # TensorBoard へ記録
        model.logger.record("episode/success_rate", ep_success)  # 0 or 1
        model.logger.record("time/virtual_steps", cumulative_virtual_steps)
        model.logger.record("time/real_elapsed", real_elapsed)
        # SAC 自動ログ（loss等）は別途 yax… 
        model.logger.dump(ep * NUM_ENVS)

        # 定期保存
        if ep % 10 == 0:
            model.save(f"{SAVE_DIR}/grasp_ep{ep:04d}")

    # 最終保存
    model.save(f"{SAVE_DIR}/grasp_final")
    print("Training complete.")
