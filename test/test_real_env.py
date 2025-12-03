import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
from utils.rl_preprocess import preprocess_and_save
from environment import MemoryEnvReal
import pickle

print("\n=== TEST 1: CSV Loading ===")
df = pd.read_csv("data/duolingo.csv")
print(df.head())
print("Rows:", len(df))

print("\n=== TEST 2: Preprocessing ===")
item_features, user_traces = preprocess_and_save("data/duolingo.csv")
print("Item features:", len(item_features))
print("User traces:", len(user_traces))

print("\n=== TEST 3: Environment Initialization ===")
env = MemoryEnvReal(item_features=item_features, user_traces=user_traces)
state = env.reset()
print("State shape:", state.shape)

available = env.get_available_actions()
print("Available actions sample:", available[:10])

print("\n=== TEST 4: Take a step ===")
action = available[0]
next_state, reward, done, info = env.step(action)
print("Reward:", reward)
print("Info:", info)

print("\nAll tests completed.")
