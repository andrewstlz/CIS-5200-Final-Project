import matplotlib.pyplot as plt
import pandas as pd

# === PASTE YOUR RESULTS BELOW ===
results_list = [
    {'episode': 0, 'recall_rate': 1.0, 'cumulative_reward': 6, 'total_reviews': 6},
    {'episode': 1, 'recall_rate': 1.0, 'cumulative_reward': 47, 'total_reviews': 47},
    {'episode': 2, 'recall_rate': 1.0, 'cumulative_reward': 28, 'total_reviews': 28},
    {'episode': 3, 'recall_rate': 1.0, 'cumulative_reward': 7, 'total_reviews': 7},
    {'episode': 4, 'recall_rate': 1.0, 'cumulative_reward': 1, 'total_reviews': 1},
    {'episode': 5, 'recall_rate': 0.9621212121212122, 'cumulative_reward': 122, 'total_reviews': 132},
    {'episode': 6, 'recall_rate': 1.0, 'cumulative_reward': 8, 'total_reviews': 8},
    {'episode': 7, 'recall_rate': 1.0, 'cumulative_reward': 16, 'total_reviews': 16},
    {'episode': 8, 'recall_rate': 1.0, 'cumulative_reward': 10, 'total_reviews': 10},
    {'episode': 9, 'recall_rate': 0.9883720930232558, 'cumulative_reward': 84, 'total_reviews': 86},
    {'episode': 10, 'recall_rate': 1.0, 'cumulative_reward': 2, 'total_reviews': 2},
    {'episode': 11, 'recall_rate': 1.0, 'cumulative_reward': 2, 'total_reviews': 2},
    {'episode': 12, 'recall_rate': 1.0, 'cumulative_reward': 7, 'total_reviews': 7},
    {'episode': 13, 'recall_rate': 1.0, 'cumulative_reward': 50, 'total_reviews': 50},
    {'episode': 14, 'recall_rate': 0.902, 'cumulative_reward': 402, 'total_reviews': 500},
    {'episode': 15, 'recall_rate': 1.0, 'cumulative_reward': 8, 'total_reviews': 8},
    {'episode': 16, 'recall_rate': 0.902, 'cumulative_reward': 402, 'total_reviews': 500},
    {'episode': 17, 'recall_rate': 1.0, 'cumulative_reward': 1, 'total_reviews': 1},
    {'episode': 18, 'recall_rate': 0.926, 'cumulative_reward': 426, 'total_reviews': 500},
    {'episode': 19, 'recall_rate': 1.0, 'cumulative_reward': 149, 'total_reviews': 149}
]

# Convert to DataFrame
df = pd.DataFrame(results_list)

# === Plot recall rate ===
plt.figure(figsize=(8, 4))
plt.plot(df["episode"], df["recall_rate"], marker="o")
plt.title("Recall Rate per Episode (Real Duolingo Environment)")
plt.xlabel("Episode")
plt.ylabel("Recall Rate")
plt.grid(True)
plt.savefig("results/recall_rate_real.png")
plt.show()

# === Plot cumulative reward ===
plt.figure(figsize=(8, 4))
plt.plot(df["episode"], df["cumulative_reward"], marker="o")
plt.title("Cumulative Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.savefig("results/reward_real.png")
plt.show()

print("Saved plots to results/ folder!")
