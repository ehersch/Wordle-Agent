"""Entropy-Guided PPO Wordle Agent.

Replicates the 3Blue1Brown information-theoretic approach as a policy prior:
exact entropy scores (expected info gain per word) are computed each step and
fed directly into the policy logits as: beta * entropy_scores + learned_correction.

The agent starts near-optimal (the entropy prior IS the 3B1B solver) and PPO
learns when to deviate from greedy entropy (e.g. multi-step planning, late-game
exploitation of likely solutions).

Usage:
    python entropy_ppo.py train [--episodes N]
    python entropy_ppo.py eval  [--model PATH] [--games N]
    python entropy_ppo.py play  [--model PATH]
    python entropy_ppo.py baseline [--games N] [--first-word WORD]
    python entropy_ppo.py validate
"""

import gym
import gym_wordle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import Counter, deque
from pathlib import Path

from gym_wordle.utils import to_english, to_array, get_words

PATTERN_CACHE = "pattern_matrix.npy"
N_PATTERNS = 243  # 3^5
POWERS_OF_3 = np.array([1, 3, 9, 27, 81], dtype=np.int64)

# Gym flag codes → our base-3 codes
# gym: right_pos=1 (green), wrong_pos=2 (yellow), wrong_char=3 (gray)
# ours: green=2, yellow=1, gray=0
GYM_TO_BASE3 = {0: 0, 1: 2, 2: 1, 3: 0}


# ──────────────────────────────────────────────────────────────
#  Pattern Matrix Precomputation
# ──────────────────────────────────────────────────────────────

def compute_pattern(guess: np.ndarray, solution: np.ndarray) -> int:
    """Compute Wordle pattern for a (guess, solution) pair.

    Replicates the gym_wordle step() flag logic exactly:
    left-to-right Counter that handles duplicate letters.

    Returns an int in [0, 242] encoding the 5-position pattern in base 3
    (green=2, yellow=1, gray=0).
    """
    flags = np.zeros(5, dtype=np.int64)
    counter = Counter()
    for i in range(5):
        ch = guess[i]
        counter[ch] += 1
        if ch == solution[i]:
            flags[i] = 2  # green
        elif counter[ch] <= (guess[i] == solution).sum():
            flags[i] = 1  # yellow
        else:
            flags[i] = 0  # gray
    return int(flags @ POWERS_OF_3)


def build_pattern_matrix(words: np.ndarray,
                         cache_path: str = PATTERN_CACHE) -> np.ndarray:
    """Precompute pattern matrix for all (guess, solution) pairs.

    Shape: (N, N) uint8 where N = len(words), entry [g, s] is the base-3
    pattern int for guessing word g when solution is s.
    Cached to disk after first computation.
    """
    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached pattern matrix from {cache_path}")
        return np.load(cache_path)

    n = len(words)
    print(f"Building pattern matrix ({n}x{n}) — this takes ~30-60 s …")
    matrix = np.zeros((n, n), dtype=np.uint8)
    for g in range(n):
        if g % 500 == 0:
            print(f"  {g}/{n} …")
        for s in range(n):
            matrix[g, s] = compute_pattern(words[g], words[s])

    np.save(cache_path, matrix)
    print(f"Pattern matrix saved to {cache_path}")
    return matrix


def validate_patterns(n_games: int = 200):
    """Verify precomputed patterns match the gym's step() output."""
    solution_words = get_words("solution")
    matrix = build_pattern_matrix(solution_words)

    env = gym.make("Wordle-v0")
    # Build action mapping: solution idx → gym action idx
    action_map = np.array([
        env.unwrapped.action_space.index_of(w) for w in solution_words
    ])

    mismatches = 0
    for game in range(n_games):
        env.reset()
        sol_idx = env.unwrapped.solution

        # Pick a random guess
        guess_idx = np.random.randint(len(solution_words))
        env_action = int(action_map[guess_idx])
        raw, _, _, _ = env.step(env_action)

        # Extract pattern from gym state
        gym_flags = raw[0][5:10]
        gym_pattern = sum(GYM_TO_BASE3[int(f)] * (3 ** i)
                         for i, f in enumerate(gym_flags))

        precomputed = int(matrix[guess_idx, sol_idx])
        if gym_pattern != precomputed:
            g_word = to_english(solution_words[guess_idx])
            s_word = to_english(solution_words[sol_idx])
            print(f"MISMATCH: guess={g_word} sol={s_word} "
                  f"gym={gym_pattern} pre={precomputed}")
            mismatches += 1

    print(f"Validation: {n_games} games, {mismatches} mismatches")
    return mismatches == 0


# ──────────────────────────────────────────────────────────────
#  Entropy Calculator
# ──────────────────────────────────────────────────────────────

class EntropyCalculator:
    """All information-theoretic computations for Wordle."""

    def __init__(self, pattern_matrix: np.ndarray):
        self.pattern_matrix = pattern_matrix
        self.n_words = pattern_matrix.shape[0]

    def filter_remaining(self, word_idx: int, pattern: int,
                         remaining: np.ndarray) -> np.ndarray:
        """Update boolean mask of remaining solutions after observing a pattern."""
        return remaining & (self.pattern_matrix[word_idx] == pattern)

    @staticmethod
    def info_gain(n_before: int, n_after: int) -> float:
        """Actual information gained: log2(n_before / n_after)."""
        if n_before <= 0 or n_after <= 0:
            return 0.0
        return np.log2(n_before / n_after)

    def expected_info_gain(self, word_idx: int,
                           remaining: np.ndarray) -> float:
        """Expected info gain (Shannon entropy of pattern distribution)."""
        n = remaining.sum()
        if n <= 1:
            return 0.0
        patterns = self.pattern_matrix[word_idx][remaining]
        counts = np.bincount(patterns, minlength=N_PATTERNS)
        probs = counts[counts > 0] / n
        return float(-np.sum(probs * np.log2(probs)))

    def all_expected_info_gains(self, remaining: np.ndarray) -> np.ndarray:
        """Expected info gain for every word. Returns shape (n_words,)."""
        n = int(remaining.sum())
        if n <= 1:
            return np.zeros(self.n_words)

        sub = self.pattern_matrix[:, remaining]  # (n_words, n_remaining)
        gains = np.zeros(self.n_words)
        for w in range(self.n_words):
            counts = np.bincount(sub[w], minlength=N_PATTERNS)
            probs = counts[counts > 0] / n
            gains[w] = -np.sum(probs * np.log2(probs))
        return gains

    def greedy_best_word(self, remaining: np.ndarray) -> int:
        """Word index that maximises expected info gain (3B1B strategy)."""
        gains = self.all_expected_info_gains(remaining)
        return int(np.argmax(gains))


# ──────────────────────────────────────────────────────────────
#  State Encoder (301-dim)
# ──────────────────────────────────────────────────────────────

class EntropyStateEncoder:
    """Encodes board state + information-theoretic features.

    Base (292): green mask 130 + yellow mask 130 + eliminated 26 + round 6
    Info (9):   remaining_frac 1 + remaining_entropy 1 + per-round gains 6
                + cumulative_info 1
    Total: 301
    """

    def __init__(self, n_words: int = 2314):
        self.feature_dim = 301
        self.n_words = n_words
        self.max_entropy = np.log2(n_words)  # ~11.18

    def encode(self, state: np.ndarray, remaining_count: int,
               info_gains_history: list) -> np.ndarray:
        # --- base features (identical to ppo.py StateEncoder) ---
        green = np.zeros((26, 5), dtype=np.float32)
        yellow = np.zeros((26, 5), dtype=np.float32)
        eliminated = np.zeros(26, dtype=np.float32)
        round_oh = np.zeros(6, dtype=np.float32)

        n_filled = 0
        for r in range(6):
            chars = state[r][:5]
            flags = state[r][5:]
            if chars[0] == 0:
                break
            n_filled += 1
            for i in range(5):
                ch = int(chars[i]) - 1
                fl = int(flags[i])
                if ch < 0:
                    continue
                if fl == 1:     # green
                    green[ch, i] = 1.0
                elif fl == 2:   # yellow
                    yellow[ch, i] = 1.0
                elif fl == 3:   # gray
                    if green[ch].sum() == 0 and yellow[ch].sum() == 0:
                        eliminated[ch] = 1.0

        round_oh[min(n_filled, 5)] = 1.0

        # --- information-theoretic features ---
        remaining_frac = np.float32(remaining_count / self.n_words)
        if remaining_count > 0:
            remaining_ent = np.float32(np.log2(remaining_count) / self.max_entropy)
        else:
            remaining_ent = np.float32(0.0)

        per_round = np.zeros(6, dtype=np.float32)
        cumulative = 0.0
        for i, g in enumerate(info_gains_history):
            if i < 6:
                per_round[i] = np.float32(g / self.max_entropy)
            cumulative += g
        cumulative_norm = np.float32(cumulative / self.max_entropy)

        return np.concatenate([
            green.flatten(), yellow.flatten(), eliminated, round_oh,
            [remaining_frac, remaining_ent],
            per_round,
            [cumulative_norm],
        ])


# ──────────────────────────────────────────────────────────────
#  Actor-Critic (same architecture as ppo.py)
# ──────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Entropy-prior policy: combines exact entropy scores with a learned
    correction via dot-product attention."""

    def __init__(self, state_dim, word_features, hidden=256, embed_dim=128):
        super().__init__()
        n_words, word_feat_dim = word_features.shape

        self.register_buffer("word_features",
                             torch.FloatTensor(word_features))

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.actor_query = nn.Linear(hidden, embed_dim)
        self.word_key = nn.Linear(word_feat_dim, embed_dim, bias=False)

        # Learnable weighting of the entropy prior (starts high to trust it)
        self.beta = nn.Parameter(torch.tensor(10.0))

        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, state, entropy_scores):
        """
        state: (batch, state_dim)
        entropy_scores: (batch, n_words) — exact expected info gain per word
        """
        h = self.state_net(state)

        # Learned correction via dot-product attention
        query = self.actor_query(h)
        keys = self.word_key(self.word_features)
        learned = query @ keys.T / (keys.shape[1] ** 0.5)

        # Combine: entropy prior + learned correction
        logits = self.beta * entropy_scores + learned

        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, state, entropy_scores, action=None):
        logits, value = self.forward(state, entropy_scores)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ──────────────────────────────────────────────────────────────
#  Rollout Buffer (same as ppo.py)
# ──────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.entropy_scores = []

    def add(self, state, action, log_prob, reward, done, value,
            entropy_scores):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.entropy_scores.append(entropy_scores)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        rewards = np.array(self.rewards)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [last_value])

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = (rewards[t] + gamma * values[t + 1] * (1 - dones[t])
                     - values[t])
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return returns.astype(np.float32), advantages.astype(np.float32)

    def iterate(self, batch_size, returns, advantages):
        n = len(self.states)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                np.array([self.states[i] for i in idx]),
                np.array([self.actions[i] for i in idx]),
                np.array([self.log_probs[i] for i in idx]),
                returns[idx],
                advantages[idx],
                np.array([self.entropy_scores[i] for i in idx]),
            )


# ──────────────────────────────────────────────────────────────
#  Wordle Wrapper with Entropy Tracking
# ──────────────────────────────────────────────────────────────

class EntropyWordleWrapper:
    """Wraps the gym env, tracks remaining solutions, computes entropy
    scores each step for use as a policy prior."""

    def __init__(self, entropy_calc: EntropyCalculator):
        self.env = gym.make("Wordle-v0")
        self.entropy_calc = entropy_calc

        solution_words = get_words("solution")
        self.solution_words = solution_words
        self.n_actions = len(solution_words)

        self.encoder = EntropyStateEncoder(self.n_actions)

        # wrapper idx → gym action idx
        self.action_map = np.array([
            self.env.unwrapped.action_space.index_of(w)
            for w in solution_words
        ])

        # Per-word letter features for dot-product attention (same as ppo.py)
        self.word_features = np.zeros((self.n_actions, 130), dtype=np.float32)
        for i, word in enumerate(solution_words):
            for pos in range(5):
                letter = int(word[pos]) - 1
                self.word_features[i, pos * 26 + letter] = 1.0

        # Episode state
        self.remaining = None
        self.info_gains_history = None
        self.entropy_scores = None  # shape (n_actions,) — expected info gain per word

    def _compute_entropy_scores(self):
        """Compute entropy scores with exploitation bonus for remaining words.

        When few words remain, entropy scores approach 0 for all words.
        The bonus ensures the agent prefers guessing remaining (viable) words,
        completing the 3B1B strategy which special-cases "1 word left → guess it".
        """
        scores = self.entropy_calc.all_expected_info_gains(
            self.remaining
        ).astype(np.float32)
        n_rem = int(self.remaining.sum())
        if n_rem > 0:
            scores += self.remaining.astype(np.float32) / n_rem
        return scores

    def reset(self):
        raw = self.env.reset()
        self.remaining = np.ones(self.n_actions, dtype=bool)
        self.info_gains_history = []
        self.entropy_scores = self._compute_entropy_scores()
        return self.encoder.encode(raw, self.n_actions, self.info_gains_history)

    def _extract_pattern(self, raw_state: np.ndarray, round_idx: int) -> int:
        """Convert gym flags to our base-3 pattern integer."""
        gym_flags = raw_state[round_idx][5:10]
        return sum(GYM_TO_BASE3[int(f)] * (3 ** i)
                   for i, f in enumerate(gym_flags))

    def step(self, wrapper_action: int):
        n_before = int(self.remaining.sum())

        env_action = int(self.action_map[wrapper_action])
        raw, gym_reward, done, info = self.env.step(env_action)

        round_idx = self.env.unwrapped.round - 1
        pattern = self._extract_pattern(raw, round_idx)

        self.remaining = self.entropy_calc.filter_remaining(
            wrapper_action, pattern, self.remaining
        )
        n_after = int(self.remaining.sum())

        # Safety: if filtering wiped everything, keep at least the solution
        if n_after == 0:
            self.remaining = np.ones(self.n_actions, dtype=bool)
            n_after = self.n_actions

        ig = EntropyCalculator.info_gain(n_before, n_after)
        self.info_gains_history.append(ig)

        # Recompute entropy scores for updated remaining set
        self.entropy_scores = self._compute_entropy_scores()

        # Win-focused reward: -1 per guess, +10 for winning
        correct = done and gym_reward == 0.0
        reward = -1.0 + (10.0 if correct else 0.0)

        encoded = self.encoder.encode(
            raw, n_after, self.info_gains_history
        )
        return encoded, reward, done, info

    @property
    def state_dim(self):
        return self.encoder.feature_dim


# ──────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────

def train(n_episodes=100_000, rollout_steps=2048, n_epochs=4,
          batch_size=256, lr=3e-4, gamma=0.99, gae_lambda=0.95,
          clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
          log_every=500, save_every=5000):

    solution_words = get_words("solution")
    matrix = build_pattern_matrix(solution_words)
    ecalc = EntropyCalculator(matrix)
    env = EntropyWordleWrapper(ecalc)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = ActorCritic(env.state_dim, env.word_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"State dim:    {env.state_dim}")
    print(f"Action dim:   {env.n_actions}")
    print(f"Device:       {device}")
    print(f"Reward:       -1/guess + 10*win (entropy prior in logits)")
    print(f"Model params: {n_params:,}")
    print(f"Initial beta: {model.beta.item():.1f}")
    print(f"Training for {n_episodes} episodes …\n")

    ep_rewards = deque(maxlen=log_every)
    ep_wins = deque(maxlen=log_every)
    ep_lengths = deque(maxlen=log_every)
    ep_info_gains = deque(maxlen=log_every)
    best_win_rate = 0.0
    total_episodes = 0
    total_steps = 0

    state = env.reset()
    ep_reward = 0.0

    while total_episodes < n_episodes:
        buffer = RolloutBuffer()

        # ---- collect rollout ----
        for _ in range(rollout_steps):
            ent_scores = env.entropy_scores  # grab before stepping

            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                e_t = torch.FloatTensor(ent_scores).unsqueeze(0).to(device)
                action, log_prob, _, value = \
                    model.get_action_and_value(s_t, e_t)

            a = action.item()
            next_state, reward, done, _ = env.step(a)
            buffer.add(state, a, log_prob.item(), reward, done,
                       value.item(), ent_scores)

            ep_reward += reward
            state = next_state
            total_steps += 1

            if done:
                uw = env.env.unwrapped
                last_guess = uw.state[uw.round - 1][:5]
                sol = uw.solution_space[uw.solution]
                correct = (last_guess == sol).all()
                ep_rewards.append(ep_reward)
                ep_wins.append(float(correct))
                ep_lengths.append(uw.round)
                ep_info_gains.append(sum(env.info_gains_history))
                total_episodes += 1
                ep_reward = 0.0
                state = env.reset()

                if (total_episodes % log_every == 0
                        and len(ep_rewards) > 0):
                    wr = np.mean(ep_wins) * 100
                    print(f"Ep {total_episodes:>7d} | "
                          f"Rew {np.mean(ep_rewards):>7.2f} | "
                          f"Win% {wr:>5.1f} | "
                          f"Len {np.mean(ep_lengths):.1f} | "
                          f"Info {np.mean(ep_info_gains):.2f}b | "
                          f"Beta {model.beta.item():.2f} | "
                          f"Steps {total_steps}")

                    if wr > best_win_rate:
                        best_win_rate = wr
                        torch.save(model.state_dict(),
                                   "entropy_ppo_best.pt")
                        print(f"  -> New best model saved ({wr:.1f}%)")

                if total_episodes >= n_episodes:
                    break

        # ---- compute GAE ----
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            e_t = torch.FloatTensor(env.entropy_scores).unsqueeze(0).to(device)
            _, _, _, last_val = model.get_action_and_value(s_t, e_t)
            last_v = 0.0 if done else last_val.item()

        returns, advantages = buffer.compute_gae(last_v, gamma, gae_lambda)
        advantages = ((advantages - advantages.mean())
                      / (advantages.std() + 1e-8))

        # ---- PPO update ----
        for _ in range(n_epochs):
            for mb in buffer.iterate(batch_size, returns, advantages):
                mb_s, mb_a, mb_olp, mb_ret, mb_adv, mb_ent = mb

                s_t = torch.FloatTensor(mb_s).to(device)
                a_t = torch.LongTensor(mb_a).to(device)
                olp_t = torch.FloatTensor(mb_olp).to(device)
                ret_t = torch.FloatTensor(mb_ret).to(device)
                adv_t = torch.FloatTensor(mb_adv).to(device)
                ent_t = torch.FloatTensor(mb_ent).to(device)

                _, new_lp, entropy, values = \
                    model.get_action_and_value(s_t, ent_t, a_t)

                ratio = torch.exp(new_lp - olp_t)
                surr1 = ratio * adv_t
                surr2 = (torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                         * adv_t)
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = F.mse_loss(values, ret_t)
                ent_loss = -entropy.mean()

                loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if total_episodes % save_every < (rollout_steps // 6 + 1):
            torch.save(model.state_dict(),
                       f"entropy_ppo_{total_episodes}.pt")

    torch.save(model.state_dict(), "entropy_ppo_final.pt")
    print(f"\nTraining done. Best win rate: {best_win_rate:.1f}%")
    print(f"Final beta: {model.beta.item():.2f}")
    return model


# ──────────────────────────────────────────────────────────────
#  Greedy Entropy Baseline (3Blue1Brown strategy)
# ──────────────────────────────────────────────────────────────

def greedy_entropy_baseline(n_games: int = 2314,
                            first_word: str = None,
                            verbose: bool = False):
    """Pure greedy max-entropy strategy — no RL, no learning."""
    solution_words = get_words("solution")
    matrix = build_pattern_matrix(solution_words)
    ecalc = EntropyCalculator(matrix)

    n_solutions = len(solution_words)

    # Compute optimal first word (cached across games)
    if first_word is not None:
        first_idx = None
        for i, w in enumerate(solution_words):
            if to_english(w) == first_word.lower():
                first_idx = i
                break
        if first_idx is None:
            print(f"Warning: '{first_word}' not in solution list, "
                  f"computing optimal first word")
            first_word = None

    if first_word is None:
        print("Computing optimal first word …")
        remaining_all = np.ones(n_solutions, dtype=bool)
        first_idx = ecalc.greedy_best_word(remaining_all)
        print(f"Optimal first word: "
              f"{to_english(solution_words[first_idx]).upper()}")

    wins = 0
    total_rounds = 0
    dist = [0] * 7  # [fail, 1, 2, 3, 4, 5, 6]

    targets = (np.random.choice(n_solutions, n_games, replace=False)
               if n_games <= n_solutions
               else np.random.choice(n_solutions, n_games, replace=True))

    for gi, target_idx in enumerate(targets):
        remaining = np.ones(n_solutions, dtype=bool)
        solved = False

        for rnd in range(6):
            if rnd == 0:
                guess_idx = first_idx
            else:
                # If only one left, guess it
                if remaining.sum() == 1:
                    guess_idx = int(np.where(remaining)[0][0])
                else:
                    guess_idx = ecalc.greedy_best_word(remaining)

            pattern = int(matrix[guess_idx, target_idx])

            if verbose:
                word = to_english(solution_words[guess_idx]).upper()
                n_rem = int(remaining.sum())
                ig = ecalc.info_gain(n_rem,
                                     int(ecalc.filter_remaining(
                                         guess_idx, pattern,
                                         remaining).sum()))
                print(f"  R{rnd+1}: {word}  pattern={pattern:>3d}  "
                      f"info={ig:.2f}b  remaining={n_rem}")

            # Check win (all green = pattern 242 = 2+6+18+54+162)
            if pattern == 242:
                solved = True
                total_rounds += rnd + 1
                dist[rnd + 1] += 1
                wins += 1
                break

            remaining = ecalc.filter_remaining(guess_idx, pattern, remaining)

        if not solved:
            total_rounds += 6
            dist[0] += 1

        if verbose:
            sol = to_english(solution_words[target_idx]).upper()
            print(f"  {'WON' if solved else 'LOST'} — target: {sol}\n")

        if (gi + 1) % 500 == 0:
            print(f"  {gi+1}/{n_games} games …")

    print(f"\nGreedy Entropy Baseline ({n_games} games):")
    print(f"  Win rate:   {wins / n_games * 100:.1f}%")
    print(f"  Avg rounds: {total_rounds / n_games:.2f}")
    print(f"  1: {dist[1]}  2: {dist[2]}  3: {dist[3]}  "
          f"4: {dist[4]}  5: {dist[5]}  6: {dist[6]}  fail: {dist[0]}")

    return {"win_rate": wins / n_games, "avg_rounds": total_rounds / n_games,
            "distribution": dist,
            "first_word": to_english(solution_words[first_idx])}


# ──────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(model_path="entropy_ppo_best.pt", n_games=100,
             compare_greedy=False):
    solution_words = get_words("solution")
    matrix = build_pattern_matrix(solution_words)
    ecalc = EntropyCalculator(matrix)
    env = EntropyWordleWrapper(ecalc)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = ActorCritic(env.state_dim, env.word_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Loaded model with beta = {model.beta.item():.2f}")

    wins = 0
    total_rounds = 0
    total_info = 0.0
    dist = [0] * 7
    entropy_agrees = 0
    total_guesses = 0

    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                e_t = torch.FloatTensor(
                    env.entropy_scores).unsqueeze(0).to(device)
                logits, _ = model(s_t, e_t)
                action = logits.argmax(1).item()

            # Track agreement with greedy entropy
            greedy_action = int(np.argmax(env.entropy_scores))
            if action == greedy_action:
                entropy_agrees += 1
            total_guesses += 1

            state, reward, done, _ = env.step(action)

        rounds = env.env.unwrapped.round
        total_rounds += rounds
        total_info += sum(env.info_gains_history)

        last_guess = env.env.unwrapped.state[rounds - 1][:5]
        sol = env.env.unwrapped.solution_space[env.env.unwrapped.solution]
        correct = (last_guess == sol).all()
        if correct:
            wins += 1
            dist[rounds] += 1
        else:
            dist[0] += 1

    agree_pct = entropy_agrees / total_guesses * 100 if total_guesses else 0
    print(f"\nRL Agent Evaluation ({n_games} games):")
    print(f"  Win rate:    {wins / n_games * 100:.1f}%")
    print(f"  Avg rounds:  {total_rounds / n_games:.2f}")
    print(f"  Avg info:    {total_info / n_games:.2f} bits")
    print(f"  Entropy agreement: {agree_pct:.1f}% "
          f"({entropy_agrees}/{total_guesses} guesses)")
    print(f"  1: {dist[1]}  2: {dist[2]}  3: {dist[3]}  "
          f"4: {dist[4]}  5: {dist[5]}  6: {dist[6]}  fail: {dist[0]}")

    if compare_greedy:
        print("\n--- Greedy baseline for comparison ---")
        greedy_entropy_baseline(n_games=n_games)


def play_one(model_path="entropy_ppo_best.pt", target_word=None):
    """Play one game, showing entropy scores and agent reasoning."""
    solution_words = get_words("solution")
    matrix = build_pattern_matrix(solution_words)
    ecalc = EntropyCalculator(matrix)
    env = EntropyWordleWrapper(ecalc)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = ActorCritic(env.state_dim, env.word_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Model beta = {model.beta.item():.2f}\n")

    state = env.reset()

    # Override solution if target word specified
    if target_word is not None:
        target_arr = to_array(target_word.lower())
        target_idx = env.env.unwrapped.solution_space.index_of(target_arr)
        if target_idx == -1:
            print(f"Warning: '{target_word}' not in solution list, using random word")
        else:
            env.env.unwrapped.solution = target_idx

    sol = to_english(
        env.env.unwrapped.solution_space[env.env.unwrapped.solution]
    )
    print(f"Target: {sol.upper()}\n")

    done = False
    rnd = 0

    while not done:
        ent_scores = env.entropy_scores
        n_rem = int(env.remaining.sum())

        # Top-5 by entropy
        top5_idx = np.argsort(ent_scores)[-5:][::-1]
        top5_str = ", ".join(
            f"{to_english(solution_words[i]).upper()} ({ent_scores[i]:.2f}b)"
            for i in top5_idx
        )

        # Greedy entropy pick
        greedy_idx = int(np.argmax(ent_scores))
        greedy_word = to_english(solution_words[greedy_idx]).upper()

        # RL agent's choice (uses entropy prior + learned correction)
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            e_t = torch.FloatTensor(ent_scores).unsqueeze(0).to(device)
            logits, _ = model(s_t, e_t)
            action = logits.argmax(1).item()

        guess = to_english(solution_words[action]).upper()
        agrees = " (= entropy)" if action == greedy_idx else \
                 f" (entropy: {greedy_word})"

        state, reward, done, _ = env.step(action)
        ig = env.info_gains_history[-1]
        n_after = int(env.remaining.sum())

        print(f"  R{rnd+1}: {guess}{agrees}")
        print(f"       info={ig:.2f}b  remaining={n_rem}→{n_after}")
        print(f"       top5: {top5_str}")
        rnd += 1

    env.env.render()

    last_guess = env.env.unwrapped.state[env.env.unwrapped.round - 1][:5]
    sol_arr = env.env.unwrapped.solution_space[env.env.unwrapped.solution]
    correct = (last_guess == sol_arr).all()
    print(f"\n{'WON' if correct else 'LOST'} in "
          f"{env.env.unwrapped.round} rounds  "
          f"(total info: {sum(env.info_gains_history):.2f} bits)")


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Information-Theoretic PPO Wordle Agent")
    parser.add_argument("mode",
                        choices=["train", "eval", "play",
                                 "baseline", "validate"])
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--model", default="entropy_ppo_best.pt")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--first-word", default=None)
    parser.add_argument("--word", default=None,
                        help="Target word for play mode (for comparison screenshots)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        train(n_episodes=args.episodes)
    elif args.mode == "eval":
        evaluate(model_path=args.model, n_games=args.games,
                 compare_greedy=True)
    elif args.mode == "play":
        play_one(model_path=args.model, target_word=args.word)
    elif args.mode == "baseline":
        greedy_entropy_baseline(n_games=args.games,
                                first_word=args.first_word,
                                verbose=args.verbose)
    elif args.mode == "validate":
        validate_patterns()
