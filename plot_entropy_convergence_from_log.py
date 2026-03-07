#!/usr/bin/env python3
"""Parse entropy ablation train logs and plot reward convergence."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


MODE_RE = re.compile(r"Training entropy PPO with reward mode:\s*([a-z_]+)")
EP_RE = re.compile(r"Ep\s+(\d+)\s+\|\s+Rew\s+(-?\d+(?:\.\d+)?)\s+\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--out", default="results_entropy_rewards/reward_convergence.png")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    mode = None
    curves = defaultdict(lambda: {"ep": [], "rew": []})

    for line in log_path.read_text().splitlines():
        mm = MODE_RE.search(line)
        if mm:
            mode = mm.group(1)
            continue

        em = EP_RE.search(line)
        if em and mode is not None:
            curves[mode]["ep"].append(int(em.group(1)))
            curves[mode]["rew"].append(float(em.group(2)))

    if not curves:
        raise RuntimeError("No train curves found in log file.")

    plt.figure(figsize=(9, 5))
    for k in sorted(curves.keys()):
        if curves[k]["ep"]:
            plt.plot(curves[k]["ep"], curves[k]["rew"], label=k)

    plt.xlabel("Episode")
    plt.ylabel("Logged mean reward")
    plt.title("Entropy PPO Reward Ablation: Reward Convergence")
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
