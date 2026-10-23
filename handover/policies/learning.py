"""Shared decision logic for the reinforcement-learning policies.

Decision points are the steps where a user has a candidate neighbour.
A transition links one decision point of a user to that user's next
decision point, so the next state really depends on the chosen action
(a handover changes the serving cell and therefore what is observed next).
"""

import numpy as np

from .base import Observation, Policy
from .rewards import paper_reward


class LearningPolicy(Policy):
    """Subclasses implement `encode`, `q_values`, `learn`, `snapshot`, `restore`."""

    def __init__(self, margin=0.0, explore_start=0.5, explore_end=0.05,
                 explore_decisions=200_000, seed=0):
        self.margin = margin                 # confidence margin used when not training
        self.explore_start, self.explore_end = explore_start, explore_end
        self.explore_decisions = explore_decisions
        self.rng = np.random.default_rng(seed)
        self.training = False
        self.decisions_seen = 0

    # --- To be implemented by subclasses ---
    def encode(self, obs: Observation, users: np.ndarray) -> np.ndarray: ...
    def q_values(self, states: np.ndarray) -> np.ndarray: ...
    def learn(self, states, actions, rewards, next_states, done) -> None: ...
    def snapshot(self): ...
    def restore(self, snapshot) -> None: ...

    # --- Shared logic ---
    def explore_rate(self) -> float:
        progress = min(self.decisions_seen / self.explore_decisions, 1.0)
        return self.explore_start + progress * (self.explore_end - self.explore_start)

    def reset(self, num_users: int) -> None:
        self.pending = None                  # last (state, action, reward) per user
        self.has_pending = np.zeros(num_users, dtype=bool)
        self.episode_reward, self.episode_decisions = 0.0, 0

    def decide(self, obs: Observation) -> np.ndarray:
        actions = np.zeros(len(obs.rsrp), dtype=bool)
        users = np.flatnonzero(obs.candidate)
        if len(users) == 0:
            return actions

        states = self.encode(obs, users)
        q = self.q_values(states)
        if self.training:
            greedy = q[:, 1] > q[:, 0]
            explore = self.rng.random(len(users)) < self.explore_rate()
            chosen = np.where(explore, self.rng.random(len(users)) < 0.5, greedy)
            self._record(users, states, chosen, paper_reward(obs.rsrp[users], chosen))
        else:
            chosen = q[:, 1] - q[:, 0] > self.margin
        actions[users] = chosen
        return actions

    def _record(self, users, states, chosen, rewards):
        if self.pending is None:
            self.pending = (np.zeros((len(self.has_pending), states.shape[1])),
                            np.zeros(len(self.has_pending), dtype=int),
                            np.zeros(len(self.has_pending)))
        pend_state, pend_action, pend_reward = self.pending

        continuing = users[self.has_pending[users]]
        if len(continuing):
            self.learn(pend_state[continuing], pend_action[continuing], pend_reward[continuing],
                       states[self.has_pending[users]], np.zeros(len(continuing), dtype=bool))
        pend_state[users], pend_action[users], pend_reward[users] = states, chosen, rewards
        self.has_pending[users] = True

        self.decisions_seen += len(users)
        self.episode_reward += rewards.sum()
        self.episode_decisions += len(users)

    def end_episode(self) -> None:
        if self.training and self.has_pending.any():
            pend_state, pend_action, pend_reward = self.pending
            last = np.flatnonzero(self.has_pending)
            self.learn(pend_state[last], pend_action[last], pend_reward[last],
                       np.zeros_like(pend_state[last]), np.ones(len(last), dtype=bool))
