#!/usr/bin/env python3
"""Trajectory-level train/val/test splits.

Splitting is at the level of whole trajectories, never individual snapshots.
Snapshot-level splitting would leak: consecutive snapshots are one advective
time unit apart and the integral correlation time of this flow is roughly
10-30 time units, so neighbouring snapshots of the same trajectory are
strongly dependent and would appear on both sides of the split.
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train", type=int, default=80)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", type=int, default=1,
                    help="shuffle trajectory ids before splitting; each "
                         "trajectory is an independent realisation, so this "
                         "only guards against any accidental ordering")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw_dir, "traj_*.npy")))
    ids = [int(os.path.basename(f)[5:8]) for f in files]
    n = len(ids)
    assert n == args.train + args.val + args.test, (
        f"{n} trajectories present but split asks for "
        f"{args.train}+{args.val}+{args.test}")

    order = np.array(ids)
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(order)

    train = sorted(int(i) for i in order[:args.train])
    val = sorted(int(i) for i in order[args.train:args.train + args.val])
    test = sorted(int(i) for i in order[args.train + args.val:])
    assert not (set(train) & set(val)) and not (set(train) & set(test)) \
        and not (set(val) & set(test)), "splits overlap"
    assert len(train) + len(val) + len(test) == n

    per = np.load(files[0], mmap_mode="r").shape[0]
    out = {
        "_policy": "trajectory-level split; no snapshot-level leakage",
        "_why": "snapshots are 1 advective time unit apart and the integral "
                "correlation time is ~10-30 time units, so snapshots within a "
                "trajectory are strongly dependent",
        "seed": args.seed,
        "shuffled": bool(args.shuffle),
        "n_trajectories": n,
        "snapshots_per_trajectory": per,
        "train": train, "val": val, "test": test,
        "n_snapshots": {"train": len(train) * per, "val": len(val) * per,
                        "test": len(test) * per},
    }
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("train", "val", "test")}, indent=2))
    print(f"train {len(train)} / val {len(val)} / test {len(test)} "
          f"trajectories -> {args.output}")


if __name__ == "__main__":
    main()
