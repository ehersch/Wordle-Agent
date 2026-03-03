"""
entropy_dqn.py

Entropy-guided DQN for Wordle.

Idea:
- Keep a DQN Q(s,a) over a reduced action space (solution words only).
- Track the remaining candidate solution set exactly using a precomputed pattern matrix.
- Use an *entropy prior* (expected info gain) to guide exploration:
    choose argmax_a [ Q(s,a) + beta * entropy_score(a) ] under epsilon-greedy.
- Optionally shape reward with information gain:
    r = -1 + win_bonus(if solved) + ig_coef * log2(n_before / n_after)

This is basically your entropy-PPO wrapper, but swapped into a DQN training loop.

Usage:
  python entropy_dqn.py train --episodes 20000
  python entropy_dqn.py eval  --model entropy_dqn_best.pt --games 200
  python entropy_dqn.py validate

Notes:
- Pattern matrix is cached to pattern_matrix_solution.npy
- Computing entropy scores for *all* actions every step is expensive.
  This file computes entropy scores on a dynamic candidate set:
    - always includes all remaining words
    - plus a fixed global top-K list (by initial entropy)
  That gives you 3B1B-like behavior without O(N^2) per step forever.

Dependencies:
  pip install gym-wordle==0.1.3 sty==1.0.6
  gym==0.19.0 recommended if you match the original repo setup.

"""

import argparse
import math
import os
import random
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym_wordle  # noqa: F401 (registers Wordle-v0)

"""

  # --- gym-wordle compatibility (no edits to site-packages) ---
try:
    import numpy as np
    from gym_wordle.wordle import WordleEnv

    def seed(self, seed=None):
        # mimic old gym seeding: set RNG on the env
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self.np_random = np.random.RandomState(seed)
        return [seed]

    WordleEnv.seed = seed
except Exception:
    pass
# -----------------------------------------------------------

"""
from gym_wordle.utils import get_words, to_english, to_array


# ----------------------------
# Pattern matrix + entropy utils
# ----------------------------

N_PATTERNS = 243  # 3^5
POWERS_OF_3 = np.array([1, 3, 9, 27, 81], dtype=np.int64)

# gym: right_pos=1 (green), wrong_pos=2 (yellow), wrong_char=3 (gray)
# ours: green=2, yellow=1, gray=0
GYM_TO_BASE3 = {0: 0, 1: 2, 2: 1, 3: 0}

DEFAULT_PATTERN_CACHE = "pattern_matrix_solution.npy"


def compute_pattern_like_gym(guess: np.ndarray, solution: np.ndarray) -> int:
    """
    Replicates the (potentially slightly non-official) flag logic used by gym_wordle's step()
    in the entropy PPO code you pasted: left-to-right Counter.
    Returns base-3 pattern int in [0, 242] with green=2, yellow=1, gray=0.
    """
    flags = np.zeros(5, dtype=np.int64)
    counter = Counter()
    for i in range(5):
        ch = guess[i]
        counter[ch] += 1
        if ch == solution[i]:
            flags[i] = 2
        elif counter[ch] <= (ch == solution).sum():
            flags[i] = 1
        else:
            flags[i] = 0
    return int(flags @ POWERS_OF_3)


def build_pattern_matrix_solution(
    solution_words: np.ndarray,
    cache_path: str = DEFAULT_PATTERN_CACHE,
) -> np.ndarray:
    """
    Precompute pattern matrix for all (guess, solution) pairs where both are solution words.
    Shape: (N, N) uint8 where entry [g, s] is base-3 pattern int.
    """
    cache = Path(cache_path)
    if cache.exists():
        return np.load(cache_path)

    n = len(solution_words)
    print(f"Building pattern matrix ({n}x{n}) — first time can take ~30-90s…")
    matrix = np.zeros((n, n), dtype=np.uint8)
    for g in range(n):
        if g % 250 == 0:
            print(f"  {g}/{n} …")
        guess = solution_words[g]
        for s in range(n):
            matrix[g, s] = compute_pattern_like_gym(guess, solution_words[s])

    np.save(cache_path, matrix)
    print(f"Saved pattern matrix to {cache_path}")
    return matrix


def validate_patterns(n_games: int = 200) -> bool:
    """Verify precomputed patterns match the gym's step() output."""
    solution_words = get_words("solution")
    matrix = build_pattern_matrix_solution(solution_words)

    env = gym.make("Wordle-v0")

    # wrapper idx → gym action idx (solution word as a guess)
    action_map = np.array(
        [env.unwrapped.action_space.index_of(w) for w in solution_words], dtype=np.int64
    )

    mismatches = 0
    for _ in range(n_games):
        env.reset()
        sol_idx = env.unwrapped.solution

        guess_idx = np.random.randint(len(solution_words))
        env_action = int(action_map[guess_idx])
        raw, _, _, _ = env.step(env_action)

        gym_flags = raw[0][5:10]
        gym_pattern = sum(
            GYM_TO_BASE3[int(f)] * (3**i) for i, f in enumerate(gym_flags)
        )
        precomputed = int(matrix[guess_idx, sol_idx])

        if gym_pattern != precomputed:
            g_word = to_english(solution_words[guess_idx]).upper()
            s_word = to_english(solution_words[sol_idx]).upper()
            print(
                f"MISMATCH guess={g_word} sol={s_word} gym={gym_pattern} pre={precomputed}"
            )
            mismatches += 1

    print(f"Validation: {n_games} games, {mismatches} mismatches")
    return mismatches == 0


class EntropyCalculator:
    def __init__(self, pattern_matrix: np.ndarray):
        self.pattern_matrix = pattern_matrix
        self.n_words = pattern_matrix.shape[0]

    def filter_remaining(
        self, guess_idx: int, pattern: int, remaining: np.ndarray
    ) -> np.ndarray:
        return remaining & (self.pattern_matrix[guess_idx] == pattern)

    @staticmethod
    def info_gain(n_before: int, n_after: int) -> float:
        if n_before <= 0 or n_after <= 0:
            return 0.0
        return float(np.log2(n_before / n_after))

    def expected_info_gain_for_word(
        self, guess_idx: int, remaining: np.ndarray
    ) -> float:
        n = int(remaining.sum())
        if n <= 1:
            return 0.0
        patterns = self.pattern_matrix[guess_idx][remaining]
        counts = np.bincount(patterns, minlength=N_PATTERNS)
        probs = counts[counts > 0] / n
        return float(-np.sum(probs * np.log2(probs)))

    def expected_info_gain_for_many(
        self, guess_indices: np.ndarray, remaining: np.ndarray
    ) -> np.ndarray:
        """
        Compute expected info gain for a subset of guesses efficiently-ish.
        Still O(len(guess_indices) * n_remaining), but manageable when guess_indices is small/moderate.
        """
        n = int(remaining.sum())
        if n <= 1:
            return np.zeros(len(guess_indices), dtype=np.float32)

        sub = self.pattern_matrix[guess_indices][:, remaining]  # (k, n_remaining)
        out = np.zeros(len(guess_indices), dtype=np.float32)
        for i in range(sub.shape[0]):
            counts = np.bincount(sub[i], minlength=N_PATTERNS)
            probs = counts[counts > 0] / n
            out[i] = float(-np.sum(probs * np.log2(probs)))
        return out


# ----------------------------
# State encoder (same “good” features as PPO)
# ----------------------------


class WordleStateEncoder:
    """
    Encodes the raw 6x10 board into a learnable vector:
      green mask 26x5 (130)
    + yellow mask 26x5 (130)
    + eliminated letters (26)
    + round one-hot (6)
    + remaining_frac (1)
    + remaining_entropy_norm (1)
    = 294 dims

    (You can add more later; keep it simple for DQN.)
    """

    def __init__(self, n_words: int):
        self.n_words = n_words
        self.max_entropy = float(np.log2(n_words))
        self.feature_dim = 130 + 130 + 26 + 6 + 1 + 1

    def encode(self, raw_state: np.ndarray, remaining_count: int) -> np.ndarray:
        green = np.zeros((26, 5), dtype=np.float32)
        yellow = np.zeros((26, 5), dtype=np.float32)
        eliminated = np.zeros(26, dtype=np.float32)
        round_oh = np.zeros(6, dtype=np.float32)

        n_filled = 0
        for r in range(6):
            chars = raw_state[r][:5]
            flags = raw_state[r][5:]
            if chars[0] == 0:
                break
            n_filled += 1
            for i in range(5):
                ch = int(chars[i]) - 1
                fl = int(flags[i])
                if ch < 0:
                    continue
                if fl == 1:  # green
                    green[ch, i] = 1.0
                elif fl == 2:  # yellow
                    yellow[ch, i] = 1.0
                elif fl == 3:  # gray
                    if green[ch].sum() == 0 and yellow[ch].sum() == 0:
                        eliminated[ch] = 1.0

        round_oh[min(n_filled, 5)] = 1.0

        remaining_frac = np.float32(remaining_count / self.n_words)
        if remaining_count > 0:
            remaining_ent = np.float32(np.log2(remaining_count) / self.max_entropy)
        else:
            remaining_ent = np.float32(0.0)

        return np.concatenate(
            [
                green.flatten(),
                yellow.flatten(),
                eliminated,
                round_oh,
                [remaining_frac, remaining_ent],
            ]
        )


# ----------------------------
# Entropy-tracking Wordle wrapper (solution-only actions)
# ----------------------------


class EntropyWordleDQNWrapper:
    """
    Action space: indices into solution_words (size N ~ 2314).
    Under the hood: map to gym action index.
    Tracks remaining candidates via pattern matrix.
    Provides:
      - encoded state vector
      - entropy scores for a candidate subset (for exploration prior)
      - info-gain reward shaping
    """

    def __init__(
        self,
        entropy_calc: EntropyCalculator,
        pattern_matrix: np.ndarray,
        solution_words: np.ndarray,
        entropy_topk: int = 256,
        include_all_remaining_in_entropy: bool = True,
        ig_coef: float = 0.3,
        win_bonus: float = 10.0,
        step_penalty: float = 1.0,
    ):
        self.env = gym.make("Wordle-v0")
        self.ecalc = entropy_calc
        self.pattern_matrix = pattern_matrix
        self.solution_words = solution_words
        self.n_actions = len(solution_words)

        self.encoder = WordleStateEncoder(self.n_actions)

        # wrapper idx -> gym action idx
        self.action_map = np.array(
            [self.env.unwrapped.action_space.index_of(w) for w in solution_words],
            dtype=np.int64,
        )

        self.entropy_topk = int(entropy_topk)
        self.include_all_remaining = bool(include_all_remaining_in_entropy)

        self.ig_coef = float(ig_coef)
        self.win_bonus = float(win_bonus)
        self.step_penalty = float(step_penalty)

        # Episode state
        self.remaining: Optional[np.ndarray] = None
        self.raw_state: Optional[np.ndarray] = None

        # Precompute a global top-K list by initial expected entropy (one-time expensive-ish).
        # We do it ONCE and reuse as a good candidate set forever.
        self.global_topk_actions = self._compute_global_topk(self.entropy_topk)

        # Will be filled each step:
        #   entropy_scores: dict[action_idx] -> score
        self.entropy_scores: Dict[int, float] = {}

    def _compute_global_topk(self, k: int) -> np.ndarray:
        # Compute exact entropies vs full remaining for a modest random subset first
        # if you want it faster; but for N=2314, computing all is okay once.
        remaining_all = np.ones(self.n_actions, dtype=bool)
        all_idx = np.arange(self.n_actions, dtype=np.int64)
        print("Computing global top-K entropy words (one-time)…")
        ent = self.ecalc.expected_info_gain_for_many(all_idx, remaining_all)
        topk = np.argsort(ent)[-k:][::-1]
        best_word = to_english(self.solution_words[int(topk[0])]).upper()
        print(f"Global best first word (entropy): {best_word}")
        return topk.astype(np.int64)

    def reset(self) -> np.ndarray:
        self.raw_state = self.env.reset()
        self.remaining = np.ones(self.n_actions, dtype=bool)
        self._update_entropy_scores()
        return self.encoder.encode(self.raw_state, int(self.remaining.sum()))

    def _extract_pattern(self, raw_state: np.ndarray, round_idx: int) -> int:
        gym_flags = raw_state[round_idx][5:10]
        return int(sum(GYM_TO_BASE3[int(f)] * (3**i) for i, f in enumerate(gym_flags)))

    def _entropy_candidate_set(self) -> np.ndarray:
        assert self.remaining is not None
        candidates = set(map(int, self.global_topk_actions.tolist()))
        if self.include_all_remaining:
            rem_idx = np.where(self.remaining)[0].tolist()
            candidates.update(map(int, rem_idx))
        return np.array(sorted(candidates), dtype=np.int64)

    def _update_entropy_scores(self) -> None:
        assert self.remaining is not None
        cand = self._entropy_candidate_set()
        scores = self.ecalc.expected_info_gain_for_many(cand, self.remaining)

        # Small exploitation bonus: prefer guessing remaining words when n_rem is small
        n_rem = int(self.remaining.sum())
        bonus = (self.remaining[cand].astype(np.float32) / max(n_rem, 1)).astype(
            np.float32
        )
        scores = scores + bonus

        self.entropy_scores = {int(a): float(s) for a, s in zip(cand, scores)}

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self.remaining is not None
        n_before = int(self.remaining.sum())

        env_action = int(self.action_map[action_idx])
        raw, gym_reward, done, info = self.env.step(env_action)

        round_idx = self.env.unwrapped.round - 1
        pattern = self._extract_pattern(raw, round_idx)

        self.remaining = self.ecalc.filter_remaining(
            action_idx, pattern, self.remaining
        )
        n_after = int(self.remaining.sum())

        if n_after == 0:
            # Safety fallback
            self.remaining = np.ones(self.n_actions, dtype=bool)
            n_after = self.n_actions

        ig = EntropyCalculator.info_gain(n_before, n_after)

        # Solve detection (gym_reward is 0.0 if correct in your env, but robustly check board)
        uw = self.env.unwrapped
        last_guess = uw.state[uw.round - 1][:5]
        sol = uw.solution_space[uw.solution]
        correct = bool((last_guess == sol).all())

        # Reward shaping:
        # -step_penalty each guess
        # +win_bonus if solved
        # +ig_coef * info_gain
        reward = (
            -self.step_penalty
            + (self.win_bonus if correct else 0.0)
            + self.ig_coef * ig
        )

        self.raw_state = raw
        self._update_entropy_scores()
        encoded = self.encoder.encode(raw, n_after)
        return encoded, float(reward), bool(done), info

    @property
    def state_dim(self) -> int:
        return self.encoder.feature_dim


# ----------------------------
# DQN
# ----------------------------


class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.capacity
        )

    def push(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        self.buf.append((s, int(a), float(r), ns, bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ns, axis=0),
            np.array(d, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ----------------------------
# Training / Evaluation
# ----------------------------


@torch.no_grad()
def evaluate_policy(
    env: EntropyWordleDQNWrapper, qnet: DQN, device: torch.device, games: int = 200
) -> Dict:
    qnet.eval()
    wins = 0
    total_return = 0.0
    total_rounds = 0
    win_rounds: List[int] = []
    loss_rounds: List[int] = []

    for _ in range(games):
        s = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            q = qnet(st).squeeze(0).cpu().numpy()
            # pure greedy
            a = int(np.argmax(q))
            s, r, done, _ = env.step(a)
            ep_ret += r

        uw = env.env.unwrapped
        rounds = int(uw.round)
        total_rounds += rounds

        last_guess = uw.state[uw.round - 1][:5]
        sol = uw.solution_space[uw.solution]
        correct = bool((last_guess == sol).all())
        if correct:
            wins += 1
            win_rounds.append(rounds)
        else:
            loss_rounds.append(rounds)

        total_return += ep_ret

    return {
        "win_rate": wins / games,
        "avg_return": total_return / games,
        "avg_rounds": total_rounds / games,
        "win_rounds": win_rounds,
        "loss_rounds": loss_rounds,
    }


def plot_guess_histogram(win_rounds: List[int], losses: int, games: int) -> None:
    plt.figure(figsize=(7, 4))
    bins = np.arange(0.5, 7.5, 1.0)
    plt.hist(win_rounds, bins=bins, edgecolor="black", alpha=0.85)
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.xlabel("Guesses to solve")
    plt.ylabel("Win count")
    plt.title("Wordle Solve Guess Histogram (Wins Only)")
    plt.figtext(
        0.5,
        0.01,
        f"games={games} wins={len(win_rounds)} losses={losses} win_rate={len(win_rounds)/games:.2%}",
        ha="center",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_training_returns(
    episode_returns: List[float],
    output_path: str = "dqn_training_rewards.png",
    show_plot: bool = False,
) -> None:
    if not episode_returns:
        print("No episode returns to plot.")
        return

    returns = np.asarray(episode_returns, dtype=np.float32)
    x = np.arange(1, len(returns) + 1)

    plt.figure(figsize=(9, 4.5))
    plt.plot(x, returns, alpha=0.35, linewidth=1.0, label="episode return")

    if len(returns) >= 100:
        kernel = np.ones(100, dtype=np.float32) / 100.0
        mean100 = np.convolve(returns, kernel, mode="valid")
        plt.plot(
            np.arange(100, len(returns) + 1),
            mean100,
            linewidth=2.0,
            label="100-episode mean",
        )

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN Training Returns")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved training return plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def train(
    episodes: int = 30000,
    buffer_size: int = 200000,
    batch_size: int = 256,
    gamma: float = 0.99,
    lr: float = 2e-4,
    target_update_tau: float = 0.01,
    start_learning: int = 2000,
    train_every: int = 1,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 200000,
    beta_start: float = 5.0,
    beta_end: float = 0.5,
    beta_decay_steps: int = 200000,
    entropy_topk: int = 256,
    ig_coef: float = 0.3,
    win_bonus: float = 10.0,
    step_penalty: float = 1.0,
    eval_every: int = 500,
    eval_games: int = 200,
    save_path: str = "entropy_dqn_best.pt",
    reward_plot_path: str = "dqn_training_rewards.png",
    show_reward_plot: bool = False,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    solution_words = get_words("solution")
    pattern = build_pattern_matrix_solution(solution_words)
    ecalc = EntropyCalculator(pattern)

    env = EntropyWordleDQNWrapper(
        entropy_calc=ecalc,
        pattern_matrix=pattern,
        solution_words=solution_words,
        entropy_topk=entropy_topk,
        include_all_remaining_in_entropy=True,
        ig_coef=ig_coef,
        win_bonus=win_bonus,
        step_penalty=step_penalty,
    )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    qnet = DQN(env.state_dim, env.n_actions).to(device)
    tgt = DQN(env.state_dim, env.n_actions).to(device)
    tgt.load_state_dict(qnet.state_dict())

    opt = optim.AdamW(qnet.parameters(), lr=lr)
    rb = ReplayBuffer(buffer_size)

    def eps_by_step(t: int) -> float:
        if t >= eps_decay_steps:
            return eps_end
        frac = t / float(eps_decay_steps)
        return eps_end + (eps_start - eps_end) * math.exp(-5.0 * frac)

    def beta_by_step(t: int) -> float:
        if t >= beta_decay_steps:
            return beta_end
        frac = t / float(beta_decay_steps)
        return beta_end + (beta_start - beta_end) * math.exp(-4.0 * frac)

    total_steps = 0
    best_wr = -1.0
    recent_returns: Deque[float] = deque(maxlen=200)
    all_returns: List[float] = []

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            eps = eps_by_step(total_steps)
            beta = beta_by_step(total_steps)

            if random.random() < eps:
                # entropy-guided random: sample from candidate set proportional to exp(beta*entropy)
                if env.entropy_scores:
                    keys = np.array(list(env.entropy_scores.keys()), dtype=np.int64)
                    vals = np.array(
                        [env.entropy_scores[k] for k in keys], dtype=np.float32
                    )
                    # softmax(beta*vals) stable
                    z = beta * (vals - vals.max())
                    p = np.exp(z)
                    p = p / p.sum()
                    a = int(np.random.choice(keys, p=p))
                else:
                    a = random.randrange(env.n_actions)
            else:
                # greedy over Q + beta*entropy_prior (only for actions we computed entropy for)
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                q = qnet(st).squeeze(0).detach().cpu().numpy()

                if env.entropy_scores:
                    for ai, sc in env.entropy_scores.items():
                        q[ai] += beta * float(sc)
                a = int(np.argmax(q))

            ns, r, done, _ = env.step(a)
            rb.push(s, a, r, ns, done)

            s = ns
            ep_ret += r
            total_steps += 1

            # learn
            if len(rb) >= start_learning and (total_steps % train_every == 0):
                bs, ba, br, bns, bd = rb.sample(batch_size)

                bs_t = torch.tensor(bs, dtype=torch.float32, device=device)
                ba_t = torch.tensor(ba, dtype=torch.int64, device=device).unsqueeze(1)
                br_t = torch.tensor(br, dtype=torch.float32, device=device)
                bns_t = torch.tensor(bns, dtype=torch.float32, device=device)
                bd_t = torch.tensor(bd, dtype=torch.float32, device=device)

                q_sa = qnet(bs_t).gather(1, ba_t).squeeze(1)

                # Double DQN target:
                with torch.no_grad():
                    next_actions = qnet(bns_t).argmax(dim=1, keepdim=True)
                    next_q = tgt(bns_t).gather(1, next_actions).squeeze(1)
                    target = br_t + gamma * (1.0 - bd_t) * next_q

                loss = nn.SmoothL1Loss()(q_sa, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
                opt.step()

                # soft update target
                with torch.no_grad():
                    for p, tp in zip(qnet.parameters(), tgt.parameters()):
                        tp.data.mul_(1.0 - target_update_tau).add_(
                            target_update_tau * p.data
                        )

        recent_returns.append(ep_ret)
        all_returns.append(ep_ret)

        if ep % 50 == 0:
            print(
                f"ep={ep:6d} steps={total_steps:9d} "
                f"mean200={np.mean(recent_returns):8.3f} "
                f"eps={eps_by_step(total_steps):.3f} beta={beta_by_step(total_steps):.3f}"
            )

        if ep % eval_every == 0:
            stats = evaluate_policy(env, qnet, device, games=eval_games)
            wr = stats["win_rate"]
            print(
                f"[EVAL] ep={ep:6d} win={wr*100:5.1f}% "
                f"avg_ret={stats['avg_return']:7.3f} avg_rounds={stats['avg_rounds']:.2f}"
            )
            if wr > best_wr:
                best_wr = wr
                torch.save(qnet.state_dict(), save_path)
                print(f"  -> saved new best to {save_path} (win={best_wr*100:.1f}%)")

    # final save
    final_path = "entropy_dqn_final.pt"
    torch.save(qnet.state_dict(), final_path)
    print(f"Training done. Best win rate: {best_wr*100:.1f}%")
    print(f"Saved final model to {final_path}")
    plot_training_returns(
        all_returns, output_path=reward_plot_path, show_plot=show_reward_plot
    )


def eval_only(
    model_path: str, games: int = 200, seed: int = 0, plot_hist: bool = False
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    solution_words = get_words("solution")
    pattern = build_pattern_matrix_solution(solution_words)
    ecalc = EntropyCalculator(pattern)

    env = EntropyWordleDQNWrapper(
        entropy_calc=ecalc,
        pattern_matrix=pattern,
        solution_words=solution_words,
        entropy_topk=256,
        include_all_remaining_in_entropy=True,
        ig_coef=0.0,  # irrelevant during eval
        win_bonus=10.0,  # irrelevant during eval
        step_penalty=1.0,  # irrelevant during eval
    )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    qnet = DQN(env.state_dim, env.n_actions).to(device)
    sd = torch.load(model_path, map_location=device)
    qnet.load_state_dict(sd)
    stats = evaluate_policy(env, qnet, device, games=games)
    print(
        f"Eval {games} games: win={stats['win_rate']*100:.1f}% "
        f"avg_return={stats['avg_return']:.3f} avg_rounds={stats['avg_rounds']:.2f}"
    )
    if plot_hist:
        losses = len(stats["loss_rounds"])
        plot_guess_histogram(stats["win_rounds"], losses, games)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "validate"])
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--model", type=str, default="entropy_dqn_best.pt")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-hist", action="store_true")
    parser.add_argument(
        "--reward-plot-path", type=str, default="dqn_training_rewards.png"
    )
    parser.add_argument("--show-reward-plot", action="store_true")

    # shaping knobs
    parser.add_argument("--ig-coef", type=float, default=0.3)
    parser.add_argument("--win-bonus", type=float, default=10.0)
    parser.add_argument("--step-penalty", type=float, default=1.0)

    # entropy / exploration knobs
    parser.add_argument("--entropy-topk", type=int, default=256)
    parser.add_argument("--beta-start", type=float, default=5.0)
    parser.add_argument("--beta-end", type=float, default=0.5)
    parser.add_argument("--beta-decay-steps", type=int, default=200000)

    # epsilon knobs
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=200000)

    args = parser.parse_args()

    if args.mode == "validate":
        ok = validate_patterns()
        raise SystemExit(0 if ok else 1)

    if args.mode == "eval":
        eval_only(
            args.model, games=args.games, seed=args.seed, plot_hist=args.plot_hist
        )
        return

    # train
    train(
        episodes=args.episodes,
        ig_coef=args.ig_coef,
        win_bonus=args.win_bonus,
        step_penalty=args.step_penalty,
        entropy_topk=args.entropy_topk,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_decay_steps=args.beta_decay_steps,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        reward_plot_path=args.reward_plot_path,
        show_reward_plot=args.show_reward_plot,
        seed=args.seed,
        save_path="entropy_dqn_best.pt",
    )


if __name__ == "__main__":
    main()
