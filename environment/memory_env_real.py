import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class MemoryEnvReal:
    """
    REAL dataset-driven RL environment for spaced repetition.
    Replays actual Duolingo review sequences.
    """

    def __init__(
        self,
        item_features: pd.DataFrame,
        user_traces: Dict[int, List[Dict]],
        time_budget: float = 500.0,
        seed: Optional[int] = None
    ):
        self.item_features = item_features.set_index("item_id")
        self.user_traces = user_traces
        self.time_budget = time_budget

        self.item_ids = list(self.item_features.index)
        self.item_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.num_items = len(self.item_ids)

        if seed is not None:
            np.random.seed(seed)

        self.reset()

    def reset(self, user_id: Optional[int] = None) -> np.ndarray:
        if user_id is None:
            self.current_user = np.random.choice(list(self.user_traces.keys()))
        else:
            self.current_user = user_id

        self.trace = self.user_traces[self.current_user]
        self.step_index = 0
        self.remaining_budget = self.time_budget

        self.state = np.zeros((self.num_items, 4))
        self.state[:, 3] = self.item_features["correctness_rate"].values

        self.last_timestamp = {item_id: None for item_id in self.item_ids}

        return self.state

    def get_available_actions(self) -> np.ndarray:
        return np.arange(self.num_items)

    def step(self, action: int):
        if self.step_index >= len(self.trace):
            return self.state, 0, True, {"done": True}

        event = self.trace[self.step_index]

        true_item_id = event["item_id"]
        correct = event["correct"]
        timestamp = event["timestamp"]
        delta = event["delta"]

        item_idx = self.item_to_idx.get(true_item_id)
        if item_idx is None:
            self.step_index += 1
            return self.state, 0, False, {"skipped": True}

        self.state[item_idx, 0] += 1
        if correct:
            self.state[item_idx, 1] += 1

        seen = self.state[item_idx, 0]
        corr = self.state[item_idx, 1]
        self.state[item_idx, 3] = corr / seen

        if self.last_timestamp[true_item_id] is None:
            self.state[item_idx, 2] = delta / 3600
        else:
            gap = (timestamp - self.last_timestamp[true_item_id]).total_seconds()
            self.state[item_idx, 2] = gap / 3600

        self.last_timestamp[true_item_id] = timestamp

        reward = +1 if correct else -1

        self.step_index += 1
        self.remaining_budget -= 1

        done = (self.remaining_budget <= 0) or (self.step_index >= len(self.trace))

        info = {
            "true_item_id": true_item_id,
            "correct": correct,
            "timestamp": timestamp,
            "step": self.step_index
        }

        return self.state, reward, done, info

    def render(self):
        print(f"User: {self.current_user}, Step: {self.step_index}/{len(self.trace)}")
        print(f"Remaining Budget: {self.remaining_budget}")
