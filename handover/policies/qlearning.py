"""Tabular Q-learning baseline (paper: RSRP and CINR binned in 5 dB steps)."""

import numpy as np

from .base import Observation
from .learning import LearningPolicy

RSRP_EDGES = np.arange(-140, -40, 5)     # 5 dB bins
CINR_EDGES = np.arange(-20, 35, 5)


class QLearningPolicy(LearningPolicy):
    name = "RL"

    def __init__(self, alpha=0.1, gamma=0.9, **kwargs):
        super().__init__(margin=0.0, **kwargs)
        self.alpha, self.gamma = alpha, gamma
        self.table = np.zeros(((len(RSRP_EDGES) + 1) * (len(CINR_EDGES) + 1), 2))

    def encode(self, obs: Observation, users: np.ndarray) -> np.ndarray:
        rsrp_bin = np.digitize(obs.rsrp[users], RSRP_EDGES)
        cinr_bin = np.digitize(obs.cinr[users], CINR_EDGES)
        return (rsrp_bin * (len(CINR_EDGES) + 1) + cinr_bin)[:, None].astype(float)

    def q_values(self, states):
        return self.table[states[:, 0].astype(int)]

    def learn(self, states, actions, rewards, next_states, done):
        for s, a, r, s2, d in zip(states[:, 0].astype(int), actions, rewards,
                                  next_states[:, 0].astype(int), done):
            target = r if d else r + self.gamma * self.table[s2].max()
            self.table[s, a] += self.alpha * (target - self.table[s, a])

    def snapshot(self):
        return self.table.copy()

    def restore(self, snapshot):
        self.table = snapshot.copy()
