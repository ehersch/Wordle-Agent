#!/usr/bin/env python3
"""Train/evaluate Wordle agents and generate comparison plots.

Runs:
- Vanilla PPO (ppo.py)
- Entropy-guided PPO (entropy_ppo.py)
- Greedy entropy baseline (no training)

Outputs:
- Individual histograms of guesses-to-solve (wins only)
- Overlaid histogram across agents
- Win-rate bar chart
- Training reward convergence plot
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import ppo
import entropy_ppo


@dataclass
class TrainCurve:
    episodes: List[int]
    rewards: List[float]


@dataclass
class EvalStats:
    name: str
    win_rate: float
    rounds_all: List[int]
    rounds_wins: List[int]
    returns: List[float]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_and_parse_training(cmd: List[str], cwd: Path) -> TrainCurve:
    """Run training command, stream logs, parse `Ep ... | Rew ...` points."""
    ep_re = re.compile(r"Ep\s+(\d+)\s+\|\s+Rew\s+(-?\d+(?:\.\d+)?)\s+\|")
    episodes: List[int] = []
    rewards: List[float] = []

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        m = ep_re.search(line)
        if m:
            episodes.append(int(m.group(1)))
            rewards.append(float(m.group(2)))

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Training command failed (exit {rc}): {' '.join(cmd)}")

    return TrainCurve(episodes=episodes, rewards=rewards)


def evaluate_vanilla_ppo(model_path: Path, n_games: int) -> EvalStats:
    env = ppo.WordleWrapper()
    device = pick_device()

    model = ppo.ActorCritic(env.state_dim, env.word_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rounds_all: List[int] = []
    rounds_wins: List[int] = []
    returns: List[float] = []

    for _ in range(n_games):
        state = env.reset()
        done = False
        ep_return = 0.0
        reward = 0.0

        while not done:
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                scores, _ = model(s_t)
                action = int(scores.argmax(1).item())
            state, reward, done, _ = env.step(action)
            ep_return += float(reward)

        rnd = int(env.env.unwrapped.round)
        rounds_all.append(rnd)
        returns.append(ep_return)
        if reward > 0:
            rounds_wins.append(rnd)

    win_rate = len(rounds_wins) / float(n_games)
    return EvalStats(
        name="Vanilla PPO",
        win_rate=win_rate,
        rounds_all=rounds_all,
        rounds_wins=rounds_wins,
        returns=returns,
    )


def evaluate_entropy_ppo(model_path: Path, n_games: int) -> EvalStats:
    solution_words = entropy_ppo.get_words("solution")
    matrix = entropy_ppo.build_pattern_matrix(solution_words)
    ecalc = entropy_ppo.EntropyCalculator(matrix)
    env = entropy_ppo.EntropyWordleWrapper(ecalc)
    device = pick_device()

    model = entropy_ppo.ActorCritic(env.state_dim, env.word_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    rounds_all: List[int] = []
    rounds_wins: List[int] = []
    returns: List[float] = []

    for _ in range(n_games):
        state = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                e_t = torch.FloatTensor(env.entropy_scores).unsqueeze(0).to(device)
                logits, _ = model(s_t, e_t)
                action = int(logits.argmax(1).item())

            state, reward, done, _ = env.step(action)
            ep_return += float(reward)

        rnd = int(env.env.unwrapped.round)
        rounds_all.append(rnd)
        returns.append(ep_return)

        last_guess = env.env.unwrapped.state[rnd - 1][:5]
        sol = env.env.unwrapped.solution_space[env.env.unwrapped.solution]
        if bool((last_guess == sol).all()):
            rounds_wins.append(rnd)

    win_rate = len(rounds_wins) / float(n_games)
    return EvalStats(
        name="Entropy PPO",
        win_rate=win_rate,
        rounds_all=rounds_all,
        rounds_wins=rounds_wins,
        returns=returns,
    )


def evaluate_greedy_entropy(n_games: int, first_word: str = None) -> EvalStats:
    solution_words = entropy_ppo.get_words("solution")
    matrix = entropy_ppo.build_pattern_matrix(solution_words)
    ecalc = entropy_ppo.EntropyCalculator(matrix)
    n_solutions = len(solution_words)

    if first_word is not None:
        first_idx = None
        for i, w in enumerate(solution_words):
            if entropy_ppo.to_english(w) == first_word.lower():
                first_idx = i
                break
        if first_idx is None:
            print(f"Warning: first word '{first_word}' not in solution list; using optimal")
            first_word = None
    else:
        first_idx = None

    if first_idx is None:
        remaining_all = np.ones(n_solutions, dtype=bool)
        first_idx = int(ecalc.greedy_best_word(remaining_all))

    # Sample targets similarly to baseline implementation.
    if n_games <= n_solutions:
        targets = np.random.choice(n_solutions, n_games, replace=False)
    else:
        targets = np.random.choice(n_solutions, n_games, replace=True)

    rounds_all: List[int] = []
    rounds_wins: List[int] = []
    returns: List[float] = []

    for target_idx in targets:
        remaining = np.ones(n_solutions, dtype=bool)
        solved = False
        used_rounds = 6

        ep_return = 0.0
        for rnd in range(6):
            if rnd == 0:
                guess_idx = first_idx
            else:
                if int(remaining.sum()) == 1:
                    guess_idx = int(np.where(remaining)[0][0])
                else:
                    guess_idx = int(ecalc.greedy_best_word(remaining))

            pattern = int(matrix[guess_idx, target_idx])
            digits = np.base_repr(pattern, base=3).zfill(5)
            n_green = digits.count("2")
            n_yellow = digits.count("1")
            ep_return += 0.25 * n_green + 0.10 * n_yellow - 0.05
            if pattern == 242:
                solved = True
                used_rounds = rnd + 1
                break
            remaining = ecalc.filter_remaining(guess_idx, pattern, remaining)

        rounds_all.append(used_rounds)
        if solved:
            rounds_wins.append(used_rounds)
        returns.append(ep_return)

    win_rate = len(rounds_wins) / float(n_games)
    return EvalStats(
        name="Greedy Entropy",
        win_rate=win_rate,
        rounds_all=rounds_all,
        rounds_wins=rounds_wins,
        returns=returns,
    )


def plot_individual_histograms(stats: List[EvalStats], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    bins = np.arange(0.5, 7.5, 1.0)

    for ax, st in zip(axes, stats):
        data = st.rounds_wins
        if len(data) > 0:
            ax.hist(data, bins=bins, edgecolor="black", alpha=0.85)
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xlabel("Guesses")
        ax.set_title(f"{st.name}\nwin={st.win_rate*100:.1f}%")

    axes[0].set_ylabel("Win count")
    fig.suptitle("Guess Histogram (Wins Only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overlay_histogram(stats: List[EvalStats], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    bins = np.arange(0.5, 7.5, 1.0)
    for st in stats:
        if len(st.rounds_wins) > 0:
            plt.hist(st.rounds_wins, bins=bins, alpha=0.35, label=st.name)
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.xlabel("Guesses")
    plt.ylabel("Win count")
    plt.title("Overlay: Guess Histogram (Wins Only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_win_rate_bars(stats: List[EvalStats], out_path: Path) -> None:
    names = [s.name for s in stats]
    wr = [100.0 * s.win_rate for s in stats]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, wr)
    plt.ylabel("Win rate (%)")
    plt.ylim(0, 100)
    plt.title("Win Rate Comparison")

    for b, v in zip(bars, wr):
        plt.text(b.get_x() + b.get_width() / 2.0, v + 1.0, f"{v:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_training_convergence(
    vanilla_curve: TrainCurve,
    entropy_curve: TrainCurve,
    greedy_eval_mean_return: float,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 5))

    if vanilla_curve.episodes:
        plt.plot(vanilla_curve.episodes, vanilla_curve.rewards, label="Vanilla PPO")
    if entropy_curve.episodes:
        plt.plot(entropy_curve.episodes, entropy_curve.rewards, label="Entropy PPO")

    # Greedy has no training updates; show as a fixed reference line.
    plt.axhline(
        y=greedy_eval_mean_return,
        linestyle="--",
        linewidth=1.8,
        label="Greedy Entropy (eval mean return)",
    )

    plt.xlabel("Episode")
    plt.ylabel("Logged mean reward")
    plt.title("Training Reward Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_summary_json(stats: List[EvalStats], out_path: Path) -> None:
    payload: Dict[str, Dict] = {}
    for s in stats:
        payload[s.name] = {
            "win_rate": s.win_rate,
            "mean_rounds_all": float(np.mean(s.rounds_all)) if s.rounds_all else None,
            "mean_rounds_wins": float(np.mean(s.rounds_wins)) if s.rounds_wins else None,
            "mean_return": float(np.mean(s.returns)) if s.returns else None,
            "wins": len(s.rounds_wins),
            "games": len(s.rounds_all),
        }

    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and compare Wordle agents")
    parser.add_argument("--episodes", type=int, default=30000,
                        help="Training episodes for vanilla PPO and entropy PPO")
    parser.add_argument("--games", type=int, default=500,
                        help="Evaluation games per agent")
    parser.add_argument("--first-word", type=str, default=None,
                        help="Optional fixed first word for greedy baseline")
    parser.add_argument("--output-dir", type=str, default="comparison_outputs")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and use existing checkpoints")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    out_dir = repo_root / args.output_dir
    ensure_dir(out_dir)

    vanilla_curve = TrainCurve([], [])
    entropy_curve = TrainCurve([], [])

    vanilla_best = repo_root / "ppo_wordle_best.pt"
    entropy_best = repo_root / "entropy_ppo_best.pt"

    if not args.skip_train:
        print("\n=== Training Vanilla PPO ===")
        vanilla_curve = run_and_parse_training(
            [sys.executable, "-u", "ppo.py", "train", "--episodes", str(args.episodes)],
            cwd=repo_root,
        )
        if not vanilla_best.exists():
            raise FileNotFoundError("Expected ppo_wordle_best.pt after training")
        shutil.copy2(vanilla_best, out_dir / "ppo_wordle_best.pt")

        print("\n=== Training Entropy PPO ===")
        entropy_curve = run_and_parse_training(
            [sys.executable, "-u", "entropy_ppo.py", "train", "--episodes", str(args.episodes)],
            cwd=repo_root,
        )
        if not entropy_best.exists():
            raise FileNotFoundError("Expected entropy_ppo_best.pt after training")
        shutil.copy2(entropy_best, out_dir / "entropy_ppo_best.pt")
    else:
        print("Skipping training; using existing checkpoints.")
        if not vanilla_best.exists() or not entropy_best.exists():
            raise FileNotFoundError(
                "--skip-train was set, but required checkpoints are missing: "
                "ppo_wordle_best.pt and/or entropy_ppo_best.pt"
            )

    print("\n=== Evaluating Vanilla PPO ===")
    st_vanilla = evaluate_vanilla_ppo(vanilla_best, args.games)
    print(f"{st_vanilla.name}: win={st_vanilla.win_rate*100:.2f}%")

    print("\n=== Evaluating Entropy PPO ===")
    st_entropy = evaluate_entropy_ppo(entropy_best, args.games)
    print(f"{st_entropy.name}: win={st_entropy.win_rate*100:.2f}%")

    print("\n=== Evaluating Greedy Entropy Baseline ===")
    st_greedy = evaluate_greedy_entropy(args.games, first_word=args.first_word)
    print(f"{st_greedy.name}: win={st_greedy.win_rate*100:.2f}%")

    all_stats = [st_vanilla, st_entropy, st_greedy]

    # Plots
    plot_individual_histograms(all_stats, out_dir / "histograms_individual.png")
    plot_overlay_histogram(all_stats, out_dir / "histograms_overlay.png")
    plot_win_rate_bars(all_stats, out_dir / "win_rates_bar.png")
    plot_training_convergence(
        vanilla_curve,
        entropy_curve,
        greedy_eval_mean_return=float(np.mean(st_greedy.returns)),
        out_path=out_dir / "training_convergence.png",
    )

    write_summary_json(all_stats, out_dir / "summary.json")

    print("\nSaved outputs:")
    print(f"  - {out_dir / 'histograms_individual.png'}")
    print(f"  - {out_dir / 'histograms_overlay.png'}")
    print(f"  - {out_dir / 'win_rates_bar.png'}")
    print(f"  - {out_dir / 'training_convergence.png'}")
    print(f"  - {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
