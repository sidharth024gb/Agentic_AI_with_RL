from stable_baselines3 import PPO
from environment import GridWorldEnv

def train_agent():
    # Initialize the environment 
    env = GridWorldEnv()

    # Initialize the PPO agent
    # 'MlpPolicy' is standard for simple state-space imputs like coordinates
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    print("Training started...")
    model.learn(total_timesteps=10000)
    print("Training Complete.")

    # Save the trained agent
    model.save("ppo_gridworld")
    print("Model saved as ppo_gridworld.zip")