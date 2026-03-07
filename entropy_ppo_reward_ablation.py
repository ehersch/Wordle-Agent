#!/usr/bin/env python3
"""Run reward-function ablations for entropy-guided PPO."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

import entropy_ppo


DEFAULT_REWARD_MODES = [
    "shaped",
    "green_only",
    "no_step_penalty",
    "info_gain",
    "sparse",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_win_rates(win_rates: Dict[str, float], out_path: Path) -> None:
    names = list(win_rates.keys())
    values = [100.0 * win_rates[n] for n in names]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, values)
    plt.ylim(0, 100)
    plt.ylabel("Win rate (%)")
    plt.title("Entropy PPO Reward Ablation: Win Rate Comparison")
    plt.xticks(rotation=15)

    for b, v in zip(bars, values):
        plt.text(
            b.get_x() + b.get_width() / 2.0,
            min(v + 1.0, 99.0),
            f"{v:.1f}%",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entropy PPO reward-function ablation runner"
    )
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument(
        "--reward-modes",
        nargs="+",
        default=DEFAULT_REWARD_MODES,
        choices=entropy_ppo.REWARD_MODES,
    )
    parser.add_argument("--output-dir", type=str, default="results_entropy_reward_ablation")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    ensure_dir(out_dir)

    win_rates: Dict[str, float] = {}
    metrics: Dict[str, Dict] = {}

    print(f"Output dir: {out_dir}")
    print(f"Reward modes: {args.reward_modes}")

    for mode in args.reward_modes:
        print("\n" + "=" * 70)
        print(f"Training entropy PPO with reward mode: {mode}")
        prefix = out_dir / f"entropy_ppo_{mode}"

        entropy_ppo.train(
            n_episodes=args.episodes,
            reward_mode=mode,
            save_prefix=str(prefix),
            rollout_steps=args.rollout_steps,
            log_every=args.log_every,
            save_every=args.save_every,
        )

        best_model = prefix.with_name(prefix.name + "_best.pt")
        if not best_model.exists():
            fallback = prefix.with_name(prefix.name + "_final.pt")
            if fallback.exists():
                best_model = fallback
            else:
                raise FileNotFoundError(
                    f"No checkpoint found for mode '{mode}': "
                    f"{prefix}_best.pt or {prefix}_final.pt"
                )

        print(f"Evaluating {mode} using model: {best_model}")
        result = entropy_ppo.evaluate(
            model_path=str(best_model),
            n_games=args.games,
            compare_greedy=False,
            reward_mode=mode,
        )

        win_rates[mode] = float(result["win_rate"])
        metrics[mode] = result

    plot_path = out_dir / "win_rate_comparison.png"
    plot_win_rates(win_rates, plot_path)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))

    print("\nSaved:")
    print(f"  - {plot_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
