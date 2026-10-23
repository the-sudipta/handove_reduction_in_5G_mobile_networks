"""Offline training of every learned policy for one scenario.

1. "Historical data": episodes run under the network's legacy A3 policy.
2. From that data: the ensemble (K-Means/PCA/Decision Tree), the input
   scalers of the DQNs, and the LSTM forecaster.
3. Q-learning, Ensemble-DQN and DQN-raw are trained by interacting with
   training episodes; as in the paper, the weights of the best episode
   (mean reward per decision) are kept.

Training seeds never overlap the evaluation seeds.
"""

import numpy as np

from .history import History, STATE_COLUMNS, cell_statistics, handover_events, rsrp_sequences
from .policies import (A3Policy, DQNPolicy, EnsembleFeatures, LSTMPolicy,
                       QLearningPolicy, RewardGreedyPolicy, train_forecaster)
from .simulator import Simulator

TRAIN_SEED_BASE = 1_000_000


def collect_history(sim: Simulator, episodes: int, users: int):
    histories = []
    for e in range(episodes):
        history = History()
        sim.run(A3Policy(sim.cfg), users, seed=TRAIN_SEED_BASE + e, recorder=history)
        histories.append(history)
    return histories


def train_by_interaction(sim: Simulator, policy, episodes: int, users: int, log=print):
    policy.training = True
    best_score, best = -np.inf, None
    for e in range(episodes):
        sim.run(policy, users, seed=TRAIN_SEED_BASE + 10_000 + e)
        score = policy.episode_reward / max(policy.episode_decisions, 1)
        if score > best_score:
            best_score, best = score, policy.snapshot()
        log(f"    {policy.name:9s} episode {e + 1:2d}/{episodes}  reward/decision {score:+.3f}"
            f"  explore {policy.explore_rate():.2f}")
    policy.restore(best)
    policy.freeze()
    return policy


def train_all(sim: Simulator, history_episodes=3, train_episodes=20, users=100, seed=0, log=print):
    """Return the trained policies, keyed by name, ready for evaluation."""
    cfg = sim.cfg
    log("  collecting historical A3 data")
    histories = collect_history(sim, history_episodes, users)
    horizon = cfg.seconds_to_steps(1.0)

    log("  fitting ensemble (K-Means + PCA + Decision Tree)")
    events = handover_events(histories, horizon)
    ensemble = EnsembleFeatures(seed=seed).fit(events, cell_statistics(histories, cfg.num_cells))

    log("  training LSTM forecaster")
    x, y = rsrp_sequences(histories, window=5, stride=5, horizon_steps=cfg.seconds_to_steps(0.5),
                          max_samples=60_000, rng=np.random.default_rng(seed))
    lstm = LSTMPolicy(train_forecaster(x, y, seed=seed), offset_db=cfg.a3_offset_db)

    raw = events[STATE_COLUMNS].to_numpy()
    cluster, p_ho = ensemble.transform(raw, events["serving"].to_numpy())
    full = np.column_stack([raw, p_ho, cluster])
    explore = dict(explore_decisions=train_episodes * users * 40)

    learners = [
        QLearningPolicy(seed=seed, **explore),
        DQNPolicy(full.mean(0), full.std(0) + 1e-6, ensemble=ensemble, seed=seed, **explore),
        DQNPolicy(raw.mean(0), raw.std(0) + 1e-6, ensemble=None, seed=seed, **explore),
    ]
    for policy in learners:
        log(f"  training {policy.name}")
        train_by_interaction(sim, policy, train_episodes, users, log)

    policies = [A3Policy(cfg), lstm, *learners, RewardGreedyPolicy()]
    return {p.name: p for p in policies}
