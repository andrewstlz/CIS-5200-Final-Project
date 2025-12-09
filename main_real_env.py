from utils.rl_preprocess import preprocess_and_save
from environment import MemoryEnvReal
from models.bandit_agent import train_bandit_agent
import pickle


# STEP 1: Preprocess dataset (only need to do once)
item_features, user_traces = preprocess_and_save("data/duolingo.csv")

# STEP 2: Initialize real dataset environment
env = MemoryEnvReal(item_features=item_features, user_traces=user_traces)

# STEP 3: Train Bandit Agent on real env
results = train_bandit_agent(env, num_episodes=20)

print("Training done.")
print(results)
