from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import GridWorldEnv
import os

USE_LLM = True
log_dir = f"./logs/{'LLM' if USE_LLM else 'Baseline'}/"
os.makedirs(log_dir, exist_ok=True)

def train_agent():
    # Initialize the environment 
    env = GridWorldEnv(use_llm=USE_LLM)
    env = Monitor(env, log_dir)

    # Initialize the PPO agent
    # 'MlpPolicy' is standard for simple state-space imputs like coordinates
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    print(f"Training started (LLM={USE_LLM})...")
    model.learn(total_timesteps=10000)

    # Save the trained agent
    model_name = f"ppo_gridworld_{'LLM' if USE_LLM else 'Baseline'}"
    model.save(f"{log_dir}{model_name}")
    print(f"Training Complete: Model saved as {model_name}.zip, Logs saved to {log_dir}")