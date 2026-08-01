import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_dissertation_results():
    baseline_file = "baseline_ppo_logs.csv"
    llm_file = "llm_ppo_logs.csv"

    if not (os.path.exists(baseline_file) and os.path.exists(llm_file)):
        print("Log files not found. Run train_ppo.py first.")
        return

    df_base = pd.read_csv(baseline_file)
    df_llm = pd.read_csv(llm_file)

    # Academic plot styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Convergence Curve: Episodes vs. Cumulative Reward
    ax1.plot(df_base['Episode'], df_base['CumulativeReward'], label='Baseline PPO', color='crimson', linestyle='--')
    ax1.plot(df_llm['Episode'], df_llm['CumulativeReward'], label='LLM-Augmented PPO (Ollama)', color='navy', linewidth=2)
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Figure 1: Training Convergence Performance')
    ax1.legend()

    # 2. Latency Overhead vs. Efficiency
    ax2.bar(['Baseline PPO', 'LLM-PPO'], [df_base['Latency'].mean(), df_llm['Latency'].mean()], color=['crimson', 'navy'])
    ax2.set_ylabel('Avg. Episode Wall-Clock Latency (s)')
    ax2.set_title('Figure 2: Computational Overhead Comparison')

    plt.tight_layout()
    plt.savefig("Training_Convergence_Comparison.png", dpi=300)
    print("Plot saved as 'Training_Convergence_Comparison.png' (300 DPI)")
    plt.show()

if __name__ == "__main__":
    plot_dissertation_results()