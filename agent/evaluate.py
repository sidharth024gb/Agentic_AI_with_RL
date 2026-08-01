import numpy as np
from finance_env import FinanceEnv
from llm_planner import LLMPlanner

def evaluate_policy(num_episodes=20, use_llm=False):
    env = FinanceEnv()
    llm = LLMPlanner() if use_llm else None
    
    successes = 0
    total_steps = []
    total_rewards = []

    print(f"\nEvaluating Policy (LLM: {use_llm})...")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        ep_reward = 0
        steps = 0

        while not (done or truncated):
            if use_llm and llm:
                priors = llm.get_action_priors(obs)
                action = int(np.argmax(priors))
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1

            if done:  # Successfully reconciled
                successes += 1

        total_steps.append(steps)
        total_rewards.append(ep_reward)

    success_rate = (successes / num_episodes) * 100
    avg_steps = np.mean(total_steps)
    avg_reward = np.mean(total_rewards)

    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Steps to Goal: {avg_steps:.2f}")
    print(f"Average Cumulative Reward: {avg_reward:.2f}\n")

if __name__ == "__main__":
    evaluate_policy(num_episodes=10, use_llm=False)
    evaluate_policy(num_episodes=10, use_llm=True)