import pandas as pd
import matplotlib.pyplot as plt

df_base = pd.read_csv("logs/Baseline/monitor.csv", skiprows=1)
df_llm = pd.read_csv("logs/LLM/monitor.csv", skiprows=1)

# Smoothing: Rewards can be noisy, so we use a rolling mean
window = 50
df_base['r_smooth'] = df_base['r'].rolling(window=window).mean()
df_llm['r_smooth'] = df_llm['r'].rolling(window=window).mean()

# Create Plot
plt.figure(figsize=(10, 5))
plt.plot(df_base.index, df_base['r_smooth'], label='RL Baseline')
plt.plot(df_llm.index, df_llm['r_smooth'], label='RL + LLM Integration')

plt.title("Training Convergence: Baseline vs LLM-Augmented Agent")
plt.xlabel("Episode")
plt.ylabel("Smooth Episode Reward")
plt.legend()
plt.grid(True)
plt.show()

results = {
    "Metric": ["Final Mean Reward", "Avg. Steps to Goal", "Training Time (s)"],
    "Baseline": [df_base["r"].mean(), df_base["l"].mean(), df_base["t"].max()],
    "RL + LLM": [df_llm["r"].mean(), df_llm["l"].mean(), df_llm["t"].mean()]
}

results_df = pd.DataFrame(results)
results_df.to_excel("logs/Model_Performance.xlsx", index=False)