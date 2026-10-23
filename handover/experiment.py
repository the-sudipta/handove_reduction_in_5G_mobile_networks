"""Evaluate trained policies over seeds and user counts, and summarise.

Every (policy, users, seed) run uses the same seed for channel and mobility,
so all policies face identical radio conditions (common random numbers).
"""

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import stats

from .simulator import Simulator

METRICS = ["handovers", "ho_per_user_min", "failure_ratio", "rlf_per_user_min",
           "ping_pong_ratio", "interruption_ms_per_user_min", "throughput_mbps"]


def _run(sim: Simulator, policy, users: int, seed: int) -> dict:
    torch.set_num_threads(1)               # one core per worker process
    metrics = sim.run(policy, users, seed)
    return {"method": policy.name, "users": users, "seed": seed, **metrics}


def evaluate(sim: Simulator, policies: dict, user_counts, seeds, n_jobs=-1) -> pd.DataFrame:
    jobs = [delayed(_run)(sim, policy, users, seed)
            for users in user_counts for seed in seeds for policy in policies.values()]
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(jobs))


def summarise(runs: pd.DataFrame, keys=("scale", "users", "method")) -> pd.DataFrame:
    """Mean and 95% confidence half-width (Student t) of every metric."""
    def ci95(x):
        return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

    grouped = runs.groupby(list(keys), observed=True)[METRICS]
    summary = grouped.mean().join(grouped.agg(ci95), rsuffix="_ci")
    return summary.reset_index()
