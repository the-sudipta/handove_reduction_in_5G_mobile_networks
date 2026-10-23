"""Faithful replication of the mother paper's simulated results (Table 4).

This script does not re-implement anything: it imports the authors'
unmodified code from ``replication/upstream`` (fetched by
``fetch_upstream.sh``) and only orchestrates it, so any difference from
the paper comes from the published code itself.

Steps
-----
1. Generate the Scale1x/2x/3x datasets with the authors' generator.
2. For every seed, run the authors' pipeline (70/30 split, ensemble fit,
   3-epoch offline DQN pre-training, 30 Monte-Carlo runs per method).
3. Save every result to ``results/replication_runs.csv`` and compare the
   Scale1x numbers against Table 4 of the paper.

Usage
-----
    python replication/run_replication.py                 # seed 42, all scales
    python replication/run_replication.py --seeds 0 1 2 3 4 --scales 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
RESULTS = HERE / "results"
METRICS = ["handovers", "failure", "delay", "throughput"]

sys.path.insert(0, str(UPSTREAM))
import generate_data as authors_data          # noqa: E402  (authors' code)
import simulate_simulated as authors_sim      # noqa: E402  (authors' code)


def dataset_path(scale, seed):
    """Generate (once) and return the authors' dataset for one scale."""
    folder = HERE / "data" / f"seed{seed}"
    path = folder / f"realistic_5G_{scale}x.csv"
    if not path.exists():
        authors_data.generate_dataset(scale, seed=seed, output_dir=str(folder))
    return path


def run_one(csv_path, seed, runs, user_counts, epochs):
    """Run the authors' main() logic for one dataset and one seed."""
    np.random.seed(seed)
    df = authors_sim.load_dataset(str(csv_path))
    split = int(0.7 * len(df))
    train = df.iloc[:split]
    test = df.iloc[split:].reset_index(drop=True)

    extractor = authors_sim.FeatureExtractor()
    extractor.fit(train)
    weights = authors_sim.ProposedMethod(extractor).train_offline(train, epochs=epochs)
    model = authors_sim.ProposedMethod(extractor, pretrained_weights=weights)

    rows = []
    for users in user_counts:
        subset = test.iloc[:min(users, len(test))].copy()
        for method in authors_sim.METHODS:
            res = authors_sim.run_parallel(
                subset, method, model=model if method == "Proposed" else None, runs=runs)
            mean = res.mean(axis=0)
            ci95 = 1.96 * res.std(axis=0) / np.sqrt(len(res))   # authors' CI formula
            row = {"users": users, "method": method}
            for i, name in enumerate(METRICS):
                row[name], row[name + "_ci"] = mean[i], ci95[i]
            rows.append(row)
    return rows


def compare_with_paper(runs_df, data_seed=42):
    """Side-by-side Scale1x table: paper value vs replicated value."""
    paper = pd.read_csv(HERE / "paper_table4.csv")
    ours = (runs_df[(runs_df.scale == 1) & (runs_df.data_seed == data_seed)]
            .groupby(["users", "method"])[METRICS].mean().reset_index())
    table = paper.merge(ours, on=["users", "method"], suffixes=("_paper", "_ours"))
    table["method"] = pd.Categorical(table["method"], categories=authors_sim.METHODS)
    return table.sort_values(["users", "method"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds for the experiment (training + Monte-Carlo).")
    parser.add_argument("--data-seed", type=int, default=42,
                        help="Seed of the dataset generator (paper: 42).")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--users", type=int, nargs="+", default=authors_sim.USER_COUNTS)
    parser.add_argument("--tag", default="replication_runs",
                        help="Name of the output CSV in results/.")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    rows = []
    for scale in args.scales:
        csv_path = dataset_path(scale, args.data_seed)
        for seed in args.seeds:
            print(f"\n=== Scale{scale}x | experiment seed {seed} ===")
            for row in run_one(csv_path, seed, args.runs, args.users, args.epochs):
                rows.append({"scale": scale, "data_seed": args.data_seed, "seed": seed, **row})

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(RESULTS / f"{args.tag}.csv", index=False)
    print(f"\nSaved {RESULTS / (args.tag + '.csv')}")

    if 1 in args.scales:
        table = compare_with_paper(runs_df, args.data_seed)
        table.to_csv(RESULTS / f"{args.tag}_vs_table4.csv", index=False)
        print("\nScale1x vs Table 4 (handovers):")
        print(table[["users", "method", "handovers_paper", "handovers_ours"]].to_string(index=False))


if __name__ == "__main__":
    main()
