"""Paper phase three: Deep Q-Network with a confidence margin.

Network: state -> 32 -> 32 -> 2 (ReLU), as in the paper. Added for a sound
DQN: standardised inputs, a target network and a Huber loss.

With `ensemble` set, the state is the paper's 8 features
[RSRP, CINR, RSRQ, distance, speed, prev CINR, p_HO, pci_cluster];
without it, only the 6 raw measurements (ablation "DQN-raw").
"""

import copy
from collections import deque

import numpy as np
import torch
from torch import nn

from ..history import STATE_COLUMNS
from .base import Observation
from .learning import LearningPolicy


def q_network(inputs: int) -> nn.Module:
    return nn.Sequential(nn.Linear(inputs, 32), nn.ReLU(),
                         nn.Linear(32, 32), nn.ReLU(),
                         nn.Linear(32, 2))


class DQNPolicy(LearningPolicy):
    def __init__(self, scaler_mean, scaler_std, ensemble=None, margin=0.2, gamma=0.9,
                 lr=1e-3, batch_size=64, buffer_size=50_000, train_every=4,
                 target_sync=500, seed=0, **kwargs):
        super().__init__(margin=margin, seed=seed, **kwargs)
        torch.manual_seed(seed)
        self.name = "Proposed" if ensemble is not None else "DQN-raw"
        self.ensemble = ensemble
        self.mean, self.std = np.asarray(scaler_mean), np.asarray(scaler_std)
        self.gamma, self.batch_size = gamma, batch_size
        self.train_every, self.target_sync = train_every, target_sync

        self.online = q_network(len(self.mean))
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.updates = self.transitions = 0

    def raw_state(self, obs: Observation, users: np.ndarray) -> np.ndarray:
        raw = np.column_stack([getattr(obs, c)[users] for c in STATE_COLUMNS])
        if self.ensemble is None:
            return raw
        cluster, p_ho = self.ensemble.transform(raw, obs.serving[users])
        return np.column_stack([raw, p_ho, cluster])

    def encode(self, obs: Observation, users: np.ndarray) -> np.ndarray:
        return (self.raw_state(obs, users) - self.mean) / self.std

    @torch.no_grad()
    def q_values(self, states):
        return self.online(torch.as_tensor(states, dtype=torch.float32)).numpy()

    def learn(self, states, actions, rewards, next_states, done):
        self.buffer.extend(zip(states, actions, rewards, next_states, done))
        self.transitions += len(states)
        while self.transitions >= self.train_every:
            self.transitions -= self.train_every
            self._update()

    def _update(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = [self.buffer[i] for i in self.rng.integers(len(self.buffer), size=self.batch_size)]
        s, a, r, s2, done = (torch.as_tensor(np.array(x), dtype=torch.float32) for x in zip(*batch))
        with torch.no_grad():
            target = r + self.gamma * (1 - done) * self.target(s2).max(dim=1).values
        q = self.online(s).gather(1, a.long()[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

    def freeze(self):
        """Drop training-only state so the policy is small enough to send to workers."""
        super().freeze()
        self.buffer.clear()
        self.target = self.optimizer = None

    def snapshot(self):
        return copy.deepcopy(self.online.state_dict())

    def restore(self, snapshot):
        self.online.load_state_dict(snapshot)
        self.target.load_state_dict(snapshot)
