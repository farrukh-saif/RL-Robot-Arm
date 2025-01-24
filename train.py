import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from robot_arm_env import RobotArmEnv
import os

# Configuration
TOTAL_TIMESTEPS = 500_000
CHECKPOINT_FREQ = 30000
EVAL_FREQ = 10000
N_EVAL_EPISODES = 10
LOAD_MODEL_PATH = "ppo_robot_arm"
SAVE_MODEL_PATH = "ppo_robot_arm1"

# Create environment
env = RobotArmEnv(render_mode=None)
eval_env = Monitor(RobotArmEnv(render_mode=None))

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path="./ppo_checkpoints/",
    name_prefix="ppo_robot_arm"
)

eval_callback = EvalCallback(
    eval_env,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False
)

# Initialize or load existing model
if os.path.exists(LOAD_MODEL_PATH + '.zip'):
    print("Loading existing model...")
    model = PPO.load(LOAD_MODEL_PATH, env=env)
    print(f"Resuming training from step {model.num_timesteps}")
else:
    print("Initializing new model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.995,
    )

# Training with progress reporting
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=False  # Continue counting from previous timesteps
    )
    # Save final model
    model.save(SAVE_MODEL_PATH)
except KeyboardInterrupt:
    print("\nTraining interrupted!")
    model.save("ppo_robot_arm_interrupted")
    print("Saved interrupted model")

env.close()
