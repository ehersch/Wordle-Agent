import gym
import gym_wordle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from gym_wordle.utils import to_english, get_words


# ──────────────────────────────────────────────────────────────
#  State Encoder
# ──────────────────────────────────────────────────────────────

class StateEncoder:
    """Encodes the raw 6x10 Wordle board into a compact feature vector.

    Features:
      - Green mask:     26 letters × 5 positions (confirmed here)    = 130
      - Yellow mask:    26 letters × 5 positions (seen yellow here)  = 130
      - Eliminated:     26 letters (globally ruled out)              = 26
      - Round one-hot:  6                                            = 6
    Total: 292
    """

    def __init__(self):
        self.feature_dim = 292

    def encode(self, state: np.ndarray) -> np.ndarray:
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
                ch = int(chars[i]) - 1  # 0-indexed letter
                fl = int(flags[i])
                if ch < 0:
                    continue
                if fl == 1:    # green
                    green[ch, i] = 1.0
                elif fl == 2:  # yellow
                    yellow[ch, i] = 1.0
                elif fl == 3:  # gray — only mark eliminated if not green/yellow elsewhere
                    if green[ch].sum() == 0 and yellow[ch].sum() == 0:
                        eliminated[ch] = 1.0

        round_oh[min(n_filled, 5)] = 1.0

        return np.concatenate([green.flatten(), yellow.flatten(),
                               eliminated, round_oh])


# ──────────────────────────────────────────────────────────────
#  Actor-Critic with Letter-Aware Word Scoring
# ──────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """The policy scores each candidate word via dot-product between a
    learned state query and learned word-letter embeddings.  This lets
    the network generalise across words that share letters instead of
    treating every word as an independent action."""

    def __init__(self, state_dim, word_features, hidden=256, embed_dim=128):
        super().__init__()
        n_words, word_feat_dim = word_features.shape

        # Fixed one-hot letter features per word (5 × 26 = 130)
        self.register_buffer("word_features",
                             torch.FloatTensor(word_features))

        # Shared state encoder
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head — query / key dot-product scoring
        self.actor_query = nn.Linear(hidden, embed_dim)
        self.word_key = nn.Linear(word_feat_dim, embed_dim, bias=False)

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, state):
        h = self.state_net(state)

        query = self.actor_query(h)                        # (B, embed)
        keys = self.word_key(self.word_features)           # (W, embed)
        scores = query @ keys.T / (keys.shape[1] ** 0.5)  # scaled dot-product

        value = self.critic(h).squeeze(-1)
        return scores, value

    def get_action_and_value(self, state, action=None):
        scores, value = self.forward(state)
        dist = Categorical(logits=scores)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value


# ──────────────────────────────────────────────────────────────
#  PPO Rollout Buffer
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

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        rewards = np.array(self.rewards)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [last_value])

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
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
            )


# ──────────────────────────────────────────────────────────────
#  Wordle Wrapper
# ──────────────────────────────────────────────────────────────

class WordleWrapper:
    """Restricts action space to the 2 314 solution words and provides
    shaped rewards."""

    def __init__(self):
        self.env = gym.make("Wordle-v0")
        self.encoder = StateEncoder()

        solution_words = get_words("solution")
        self.n_actions = len(solution_words)

        # wrapper idx → gym env action idx
        self.action_map = np.array([
            self.env.unwrapped.action_space.index_of(w)
            for w in solution_words
        ])

        # Precompute per-word letter features (5 positions × 26 letters)
        self.word_features = np.zeros((self.n_actions, 130), dtype=np.float32)
        for i, word in enumerate(solution_words):
            for pos in range(5):
                letter = int(word[pos]) - 1
                self.word_features[i, pos * 26 + letter] = 1.0

    def reset(self):
        return self.encoder.encode(self.env.reset())

    def step(self, wrapper_action):
        env_action = int(self.action_map[wrapper_action])
        raw, reward, done, info = self.env.step(env_action)

        # Reward shaping: −1 per guess, +10 for a win
        shaped = -1.0
        if done and reward == 0.0:   # env gives 0 for correct guess
            shaped += 10.0

        return self.encoder.encode(raw), shaped, done, info

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

    env = WordleWrapper()

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
    print(f"Model params: {n_params:,}")
    print(f"Training for {n_episodes} episodes …\n")

    ep_rewards = deque(maxlen=log_every)
    ep_wins = deque(maxlen=log_every)
    ep_lengths = deque(maxlen=log_every)
    best_win_rate = 0.0
    total_episodes = 0
    total_steps = 0

    state = env.reset()
    ep_reward = 0.0

    while total_episodes < n_episodes:
        buffer = RolloutBuffer()

        # ---- collect rollout ----
        for _ in range(rollout_steps):
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, log_prob, _, value = model.get_action_and_value(s_t)

            a = action.item()
            next_state, reward, done, _ = env.step(a)
            buffer.add(state, a, log_prob.item(), reward, done, value.item())

            ep_reward += reward
            state = next_state
            total_steps += 1

            if done:
                won = reward > 0
                ep_rewards.append(ep_reward)
                ep_wins.append(float(won))
                ep_lengths.append(env.env.unwrapped.round)
                total_episodes += 1
                ep_reward = 0.0
                state = env.reset()

                if total_episodes % log_every == 0 and len(ep_rewards) > 0:
                    wr = np.mean(ep_wins) * 100
                    print(f"Ep {total_episodes:>7d} | "
                          f"Rew {np.mean(ep_rewards):>7.2f} | "
                          f"Win% {wr:>5.1f} | "
                          f"Len {np.mean(ep_lengths):.1f} | "
                          f"Steps {total_steps}")

                    if wr > best_win_rate:
                        best_win_rate = wr
                        torch.save(model.state_dict(), "ppo_wordle_best.pt")
                        print(f"  -> New best model saved ({wr:.1f}%)")

                if total_episodes >= n_episodes:
                    break

        # ---- compute GAE ----
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, _, _, last_val = model.get_action_and_value(s_t)
            last_v = 0.0 if done else last_val.item()

        returns, advantages = buffer.compute_gae(last_v, gamma, gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- PPO update ----
        for _ in range(n_epochs):
            for mb in buffer.iterate(batch_size, returns, advantages):
                mb_s, mb_a, mb_olp, mb_ret, mb_adv = mb

                s_t = torch.FloatTensor(mb_s).to(device)
                a_t = torch.LongTensor(mb_a).to(device)
                olp_t = torch.FloatTensor(mb_olp).to(device)
                ret_t = torch.FloatTensor(mb_ret).to(device)
                adv_t = torch.FloatTensor(mb_adv).to(device)

                _, new_lp, entropy, values = model.get_action_and_value(s_t, a_t)

                ratio = torch.exp(new_lp - olp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = F.mse_loss(values, ret_t)
                ent_loss = -entropy.mean()

                loss = pi_loss + vf_coef * v_loss + ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if total_episodes % save_every < (rollout_steps // 6 + 1):
            torch.save(model.state_dict(), f"ppo_wordle_{total_episodes}.pt")

    torch.save(model.state_dict(), "ppo_wordle_final.pt")
    print(f"\nTraining done. Best win rate: {best_win_rate:.1f}%")
    return model


# ──────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(model_path="ppo_wordle_best.pt", n_games=100):
    env = WordleWrapper()
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = ActorCritic(env.state_dim, env.word_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    solution_words = get_words("solution")
    wins = 0
    total_rounds = 0
    dist = [0] * 7  # [fail, 1, 2, 3, 4, 5, 6]

    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                scores, _ = model(s_t)
                action = scores.argmax(1).item()
            state, reward, done, _ = env.step(action)

        rounds = env.env.unwrapped.round
        total_rounds += rounds
        if reward > 0:
            wins += 1
            dist[rounds] += 1
        else:
            dist[0] += 1

    print(f"\nEvaluation ({n_games} games):")
    print(f"  Win rate:   {wins / n_games * 100:.1f}%")
    print(f"  Avg rounds: {total_rounds / n_games:.2f}")
    print(f"  1: {dist[1]}  2: {dist[2]}  3: {dist[3]}  "
          f"4: {dist[4]}  5: {dist[5]}  6: {dist[6]}  fail: {dist[0]}")


def play_one(model_path="ppo_wordle_best.pt", target_word=None):
    env = WordleWrapper()
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = ActorCritic(env.state_dim, env.word_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    solution_words = get_words("solution")
    state = env.reset()

    # Override solution if target word specified
    if target_word is not None:
        from gym_wordle.utils import to_array
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
    while not done:
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            scores, _ = model(s_t)
            action = scores.argmax(1).item()

        guess = to_english(solution_words[action])
        print(f"  Guess: {guess.upper()}")
        state, reward, done, _ = env.step(action)

    env.env.render()
    print(f"\n{'WON' if reward > 0 else 'LOST'} in "
          f"{env.env.unwrapped.round} rounds")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPO Wordle Agent")
    parser.add_argument("mode", choices=["train", "eval", "play"])
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--model", default="ppo_wordle_best.pt")
    parser.add_argument("--word", default=None,
                        help="Target word for play mode (for comparison screenshots)")
    args = parser.parse_args()

    {"train": lambda: train(n_episodes=args.episodes),
     "eval":  lambda: evaluate(model_path=args.model),
     "play":  lambda: play_one(model_path=args.model, target_word=args.word)}[args.mode]()
