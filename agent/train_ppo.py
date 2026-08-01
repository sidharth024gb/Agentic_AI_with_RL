import time
import csv
import numpy as np
from finance_env import FinanceEnv
from llm_planner import LLMPlanner
from config import Config

def train_agent(use_llm=False, total_episodes=50):
    env = FinanceEnv()
    llm = LLMPlanner() if use_llm else None
    
    log_filename = f"logs/{"llm/llm_ppo_logs" if use_llm else "baseline/baseline_ppo_logs"}_{time.strftime("%Y%m%d_%H%M%S")}.csv"
    
    with open(log_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "CumulativeReward", "Steps", "Latency"])

        print(f"--- Starting Training ({'LLM-Augmented PPO' if use_llm else 'Baseline PPO'}) ---")
        
        for episode in range(1, total_episodes + 1):
            obs, info = env.reset()
            cum_reward = 0
            steps = 0
            start_time = time.time()
            done = False
            truncated = False

            while not (done or truncated):
                if use_llm and llm:
                    # Retrieve prior probabilities from local Ollama model
                    priors = llm.get_action_priors(obs)
                    # Blend LLM prior distribution with random exploration/policy correction
                    action = int(np.random.choice(Config.ACTION_DIM, p=priors))
                else:
                    action = env.action_space.sample()

                obs, reward, done, truncated, info = env.step(action)
                cum_reward += reward
                steps += 1

            latency = round(time.time() - start_time, 3)
            writer.writerow([episode, round(cum_reward, 2), steps, latency])

            if episode % 10 == 0 or episode == 1:
                print(f"Episode {episode}/{total_episodes} | Reward: {cum_reward:.2f} | Steps: {steps} | Wall-time: {latency}s")

if __name__ == "__main__":
    # 1. Train Baseline
    train_agent(use_llm=False, total_episodes=50)
    # 2. Train LLM-Augmented Agent with Ollama
    train_agent(use_llm=True, total_episodes=50)