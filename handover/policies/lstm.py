"""LSTM baseline: predict the neighbour-minus-serving RSRP and hand over early.

A one-layer LSTM (32 units) reads the last 5 RSRP samples (one every 200 ms)
of the serving cell and the candidate, and predicts their difference 0.5 s
ahead. It hands over when the predicted difference exceeds the A3 offset,
i.e. a predictive A3 without the Time-To-Trigger wait.
"""

from collections import deque

import numpy as np
import torch
from torch import nn

from .base import Observation, Policy


class RSRPForecaster(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(1)


def train_forecaster(x, y, epochs=8, batch_size=256, lr=1e-3, seed=0):
    """Fit the forecaster with MSE; x: (n, window, 2), y: (n,)."""
    torch.manual_seed(seed)
    model = RSRPForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = torch.as_tensor(x), torch.as_tensor(y)
    for _ in range(epochs):
        for batch in torch.randperm(len(x)).split(batch_size):
            loss = nn.functional.mse_loss(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


class LSTMPolicy(Policy):
    name = "LSTM"

    def __init__(self, model: RSRPForecaster, offset_db: float, window=5, stride=5):
        self.model, self.offset_db = model, offset_db
        self.window, self.stride = window, stride

    def reset(self, num_users: int) -> None:
        self.recent = deque(maxlen=(self.window - 1) * self.stride + 1)   # RSRP of all cells

    @torch.no_grad()
    def decide(self, obs: Observation) -> np.ndarray:
        self.recent.append(obs.rsrp_all)
        actions = np.zeros(len(obs.rsrp), dtype=bool)
        users = np.flatnonzero(obs.candidate)
        if len(self.recent) < self.recent.maxlen or len(users) == 0:
            return actions

        past = np.stack(list(self.recent)[::self.stride])            # (window, users, cells)
        seq = np.stack([past[:, users, obs.serving[users]],
                        past[:, users, obs.target[users]]], axis=2)   # (window, n, 2)
        seq = np.transpose(seq - seq[-1:, :, :1], (1, 0, 2))           # relative to serving now
        predicted = self.model(torch.as_tensor(seq, dtype=torch.float32)).numpy()
        actions[users] = predicted > self.offset_db
        return actions
