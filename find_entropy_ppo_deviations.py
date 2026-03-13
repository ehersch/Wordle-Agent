#!/usr/bin/env python3
"""Find and inspect where entropy-PPO deviates from greedy entropy."""

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

import entropy_ppo
from gym_wordle.utils import get_words, to_english


@dataclass
class DeviationEvent:
    game_idx: int
    target_word: str
    round_idx: int
    remaining_before: int
    model_word: str
    greedy_word: str
    model_logit: float
    greedy_logit: float
    logit_gap: float
    solved: bool
    rounds_used: int


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_analysis(model_path: str, games: int, seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    solution_words = get_words("solution")
    matrix = entropy_ppo.build_pattern_matrix(solution_words)
    ecalc = entropy_ppo.EntropyCalculator(matrix)
    env = entropy_ppo.EntropyWordleWrapper(ecalc)

    device = pick_device()
    model = entropy_ppo.ActorCritic(env.state_dim, env.word_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    total_guesses = 0
    total_deviations = 0
    all_events: List[DeviationEvent] = []
    first_dev_per_game: List[DeviationEvent] = []

    for gi in range(1, games + 1):
        state = env.reset()
        target_idx = int(env.env.unwrapped.solution)
        target_word = to_english(solution_words[target_idx])
        done = False
        round_idx = 0
        ep_events: List[Dict] = []
        solved = False

        while not done:
            round_idx += 1
            remaining_before = int(env.remaining.sum())

            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                e_t = torch.FloatTensor(env.entropy_scores).unsqueeze(0).to(device)
                logits, _ = model(s_t, e_t)
                scores = logits.squeeze(0).cpu().numpy()

            model_action = int(np.argmax(scores))
            greedy_action = int(np.argmax(env.entropy_scores))

            total_guesses += 1
            if model_action != greedy_action:
                total_deviations += 1
                ep_events.append(
                    {
                        "round_idx": round_idx,
                        "remaining_before": remaining_before,
                        "model_word": to_english(solution_words[model_action]),
                        "greedy_word": to_english(solution_words[greedy_action]),
                        "model_logit": float(scores[model_action]),
                        "greedy_logit": float(scores[greedy_action]),
                    }
                )

            state, _, done, info = env.step(model_action)
            if done:
                solved = bool(info.get("solved", False))

        for ev in ep_events:
            event = DeviationEvent(
                game_idx=gi,
                target_word=target_word,
                round_idx=ev["round_idx"],
                remaining_before=ev["remaining_before"],
                model_word=ev["model_word"],
                greedy_word=ev["greedy_word"],
                model_logit=ev["model_logit"],
                greedy_logit=ev["greedy_logit"],
                logit_gap=ev["model_logit"] - ev["greedy_logit"],
                solved=solved,
                rounds_used=round_idx,
            )
            all_events.append(event)

        if ep_events:
            first = ep_events[0]
            first_dev_per_game.append(
                DeviationEvent(
                    game_idx=gi,
                    target_word=target_word,
                    round_idx=first["round_idx"],
                    remaining_before=first["remaining_before"],
                    model_word=first["model_word"],
                    greedy_word=first["greedy_word"],
                    model_logit=first["model_logit"],
                    greedy_logit=first["greedy_logit"],
                    logit_gap=first["model_logit"] - first["greedy_logit"],
                    solved=solved,
                    rounds_used=round_idx,
                )
            )

    deviation_pct = 100.0 * total_deviations / max(total_guesses, 1)
    return {
        "total_guesses": total_guesses,
        "total_deviations": total_deviations,
        "deviation_pct": deviation_pct,
        "all_events": all_events,
        "first_dev_per_game": first_dev_per_game,
    }


def print_examples(events: List[DeviationEvent], max_examples: int) -> None:
    if not events:
        print("No deviations found in sampled games.")
        return

    print(f"\nShowing {min(max_examples, len(events))} deviation examples:")
    for ev in events[:max_examples]:
        outcome = "WIN" if ev.solved else "LOSS"
        print(
            f"- game={ev.game_idx} target={ev.target_word.upper()} "
            f"round={ev.round_idx} remaining={ev.remaining_before} "
            f"model={ev.model_word.upper()} greedy={ev.greedy_word.upper()} "
            f"logit_gap={ev.logit_gap:+.4f} outcome={outcome} rounds={ev.rounds_used}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="entropy_ppo_best.pt")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument(
        "--replay-first",
        action="store_true",
        help="Replay the first deviation game with entropy_ppo.play_one",
    )
    args = parser.parse_args()

    out = run_analysis(model_path=args.model, games=args.games, seed=args.seed)
    print(f"Model: {args.model}")
    print(
        f"Deviation rate: {out['deviation_pct']:.2f}% "
        f"({out['total_deviations']}/{out['total_guesses']} guesses)"
    )
    print_examples(out["first_dev_per_game"], args.max_examples)

    if args.replay_first and out["first_dev_per_game"]:
        target = out["first_dev_per_game"][0].target_word
        print(
            f"\nReplaying first deviation example on target {target.upper()} "
            f"with entropy_ppo.play_one...\n"
        )
        entropy_ppo.play_one(model_path=args.model, target_word=target)


if __name__ == "__main__":
    main()
