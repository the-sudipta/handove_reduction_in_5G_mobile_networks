"""Corrected re-implementation: train and evaluate all policies on Scale1x-3x.

Outputs (in results/corrected/):
  runs.csv      one row per (scale, users, seed, method)
  summary.csv   mean and 95% CI per (scale, users, method)
  training.log  training progress

Usage
-----
    python scripts/run_corrected.py                        # full study
    python scripts/run_corrected.py --scales 1 --seeds 3 --train-episodes 4   # quick check
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                        # noqa: E402

from handover.config import scale_config                   # noqa: E402
from handover.experiment import evaluate, summarise        # noqa: E402
from handover.simulator import Simulator, make_layout      # noqa: E402
from handover.training import train_all                    # noqa: E402

ORDER = ["A3", "LSTM", "RL", "Proposed", "DQN-raw", "Eq4-greedy"]


def main():
    parser = argparse.ArgumentParser(description="Corrected handover study")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--users", type=int, nargs="+", default=[100, 200, 300, 500])
    parser.add_argument("--seeds", type=int, default=30, help="evaluation seeds per point")
    parser.add_argument("--history-episodes", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--train-users", type=int, default=100)
    parser.add_argument("--out", default=str(ROOT / "results" / "corrected"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "training.log", "w", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")

    all_runs = []
    for scale in args.scales:
        cfg = scale_config(scale)
        sim = Simulator(cfg, make_layout(cfg, layout_seed=scale))
        log(f"=== Scale{scale}x: {cfg.num_cells} cells, {cfg.area_m:.0f} m x {cfg.area_m:.0f} m ===")

        start = time.time()
        policies = train_all(sim, args.history_episodes, args.train_episodes, args.train_users, log=log)
        log(f"  training done in {time.time() - start:.0f} s; evaluating")

        start = time.time()
        runs = evaluate(sim, policies, args.users, range(args.seeds))
        runs.insert(0, "scale", scale)
        all_runs.append(runs)
        log(f"  evaluation done in {time.time() - start:.0f} s")

    runs = pd.concat(all_runs, ignore_index=True)
    runs["method"] = pd.Categorical(runs["method"], categories=ORDER)
    runs.sort_values(["scale", "users", "method", "seed"]).to_csv(out / "runs.csv", index=False)
    summary = summarise(runs)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    log("\n" + summary[["scale", "users", "method", "ho_per_user_min", "failure_ratio",
                        "ping_pong_ratio", "throughput_mbps"]].to_string(index=False))


if __name__ == "__main__":
    main()
