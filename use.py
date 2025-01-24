from stable_baselines3 import PPO
from robot_arm_env import RobotArmEnv

NUMBER_OF_EPISODES = 5

model = PPO.load("ppo_robot_arm")
print(f"\nTesting model: Trained for {model.num_timesteps}")
env = RobotArmEnv(render_mode="human")

for _ in range(NUMBER_OF_EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            break
    print(f"Test episode reward: {total_reward:.1f}")

env.close()