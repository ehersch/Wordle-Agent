"""
Reward Function Ablation Experiment for PPO Wordle Agent.

Trains multiple PPO agents with different reward function variants,
then generates:
  1. Reward convergence plot (smoothed reward over episodes)
  2. Histogram of number of guesses
  3. Win rate bar chart comparison

Usage:
  python reward_experiment.py train --episodes 50000
  python reward_experiment.py plot
  python reward_experiment.py all --episodes 50000   # train + plot
"""

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
import json, os, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Reuse model classes from ppo.py ──────────────────────────

class StateEncoder:
    def __init__(self, use_eliminated=True):
        self.use_eliminated = use_eliminated
        self.feature_dim = 292 if use_eliminated else 266  # 130+130+26+6 vs 130+130+6

    def encode(self, state):
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
                if fl == 1:
                    green[ch, i] = 1.0
                elif fl == 2:
                    yellow[ch, i] = 1.0
                elif fl == 3:
                    if green[ch].sum() == 0 and yellow[ch].sum() == 0:
                        eliminated[ch] = 1.0
        round_oh[min(n_filled, 5)] = 1.0
        if self.use_eliminated:
            return np.concatenate([green.flatten(), yellow.flatten(), eliminated, round_oh])
        else:
            return np.concatenate([green.flatten(), yellow.flatten(), round_oh])


class ActorCritic(nn.Module):
    def __init__(self, state_dim, word_features, hidden=256, embed_dim=128):
        super().__init__()
        n_words, word_feat_dim = word_features.shape
        self.register_buffer("word_features", torch.FloatTensor(word_features))
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor_query = nn.Linear(hidden, embed_dim)
        self.word_key = nn.Linear(word_feat_dim, embed_dim, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, state):
        h = self.state_net(state)
        query = self.actor_query(h)
        keys = self.word_key(self.word_features)
        scores = query @ keys.T / (keys.shape[1] ** 0.5)
        value = self.critic(h).squeeze(-1)
        return scores, value

    def get_action_and_value(self, state, action=None):
        scores, value = self.forward(state)
        dist = Categorical(logits=scores)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

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
                returns[idx], advantages[idx],
            )


# ── Reward function variants ─────────────────────────────────

REWARD_CONFIGS = {
    "baseline": {
        "green_w": 0.2, "yellow_w": 0.05,
        "gray_penalty": 0.0, "yellow_repeat_penalty": 0.0,
        "step_penalty": 0.02,
        "label": "Baseline",
    },
    "penalize_gray": {
        "green_w": 0.2, "yellow_w": 0.05,
        "gray_penalty": 0.05, "yellow_repeat_penalty": 0.0,
        "step_penalty": 0.02,
        "label": "Repeated Gray Penalty",
    },
    "penalize_yellow": {
        "green_w": 0.2, "yellow_w": 0.05,
        "gray_penalty": 0.0, "yellow_repeat_penalty": 0.05,
        "step_penalty": 0.02,
        "label": "Repeated Yellow Penalty",
    },
    "green_heavy": {
        "green_w": 0.4, "yellow_w": 0.02,
        "gray_penalty": 0.0, "yellow_repeat_penalty": 0.0,
        "step_penalty": 0.02,
        "label": "Higher Green / Lower Yellow",
    },
    "step_heavy": {
        "green_w": 0.2, "yellow_w": 0.05,
        "gray_penalty": 0.0, "yellow_repeat_penalty": 0.0,
        "step_penalty": 0.10,
        "label": "Increased Step Penalty",
    },
}


class WordleWrapperExperiment:
    def __init__(self, reward_config, use_eliminated=True):
        self.env = gym.make("Wordle-v0")
        self.encoder = StateEncoder(use_eliminated=use_eliminated)
        self.cfg = reward_config

        solution_words = get_words("solution")
        self.n_actions = len(solution_words)
        self.action_map = np.array([
            self.env.unwrapped.action_space.index_of(w) for w in solution_words
        ])
        self.solution_words = solution_words
        self.word_features = np.zeros((self.n_actions, 130), dtype=np.float32)
        for i, word in enumerate(solution_words):
            for pos in range(5):
                letter = int(word[pos]) - 1
                self.word_features[i, pos * 26 + letter] = 1.0

    def reset(self):
        self.known_gray = set()
        self.known_yellow = set()  # tracks (letter, position) pairs seen as yellow
        return self.encoder.encode(self.env.reset())

    def step(self, wrapper_action):
        env_action = int(self.action_map[wrapper_action])
        raw, _, done, info = self.env.step(env_action)

        round_idx = self.env.unwrapped.round - 1
        chars = raw[round_idx][:5]
        flags = raw[round_idx][5:10]
        right_pos = getattr(self.env.unwrapped, "right_pos", 1)
        wrong_pos = getattr(self.env.unwrapped, "wrong_pos", 2)

        n_green = int((flags == right_pos).sum())
        n_yellow = int((flags == wrong_pos).sum())

        repeat_gray = 0
        repeat_yellow = 0
        for i in range(5):
            ch = int(chars[i]) - 1
            fl = int(flags[i])
            if fl == wrong_pos:
                # Yellow: penalize if same letter was yellow before (not promoted to green)
                if ch in self.known_yellow:
                    repeat_yellow += 1
                else:
                    self.known_yellow.add(ch)
            elif fl != right_pos:
                # Gray
                if ch in self.known_gray:
                    repeat_gray += 1
                else:
                    self.known_gray.add(ch)

        c = self.cfg
        shaped = (c["green_w"] * n_green
                  + c["yellow_w"] * n_yellow
                  - c["gray_penalty"] * min(repeat_gray, 2)
                  - c["yellow_repeat_penalty"] * min(repeat_yellow, 2)
                  - c["step_penalty"])

        if done and n_green == 5:
            guesses_used = self.env.unwrapped.round
            shaped += 3.0 + (6 - guesses_used)

        return self.encoder.encode(raw), shaped, done, info

    @property
    def state_dim(self):
        return self.encoder.feature_dim


# ── Training with logging ────────────────────────────────────

def train_variant(variant_name, n_episodes=50_000, rollout_steps=2048,
                  n_epochs=4, batch_size=256, lr=3e-4, gamma=0.99,
                  gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
                  log_every=500, use_eliminated=True,
                  model_dir="models", result_dir="results"):

    cfg = REWARD_CONFIGS[variant_name]
    elim_tag = "" if use_eliminated else " (no eliminated vector)"
    print(f"\n{'='*60}")
    print(f"Training variant: {cfg['label']}{elim_tag}")
    print(f"{'='*60}")

    env = WordleWrapperExperiment(cfg, use_eliminated=use_eliminated)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = ActorCritic(env.state_dim, env.word_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Device: {device} | Episodes: {n_episodes}")

    ep_rewards = deque(maxlen=log_every)
    ep_wins = deque(maxlen=log_every)
    ep_lengths = deque(maxlen=log_every)
    best_win_rate = 0.0
    total_episodes = 0
    total_steps = 0

    # Logging
    log = {"episodes": [], "avg_reward": [], "win_rate": [], "avg_guesses": []}

    state = env.reset()
    ep_reward = 0.0

    while total_episodes < n_episodes:
        buffer = RolloutBuffer()

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
                    avg_r = np.mean(ep_rewards)
                    avg_g = np.mean(ep_lengths)

                    log["episodes"].append(total_episodes)
                    log["avg_reward"].append(float(avg_r))
                    log["win_rate"].append(float(wr))
                    log["avg_guesses"].append(float(avg_g))

                    print(f"Ep {total_episodes:>7d} | "
                          f"Rew {avg_r:>7.2f} | "
                          f"Win% {wr:>5.1f} | "
                          f"Len {avg_g:.1f}")

                    if wr > best_win_rate:
                        best_win_rate = wr
                        torch.save(model.state_dict(),
                                   f"{model_dir}/ppo_{variant_name}_best.pt")

                if total_episodes >= n_episodes:
                    break

        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, _, _, last_val = model.get_action_and_value(s_t)
            last_v = 0.0 if done else last_val.item()

        returns, advantages = buffer.compute_gae(last_v, gamma, gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

    torch.save(model.state_dict(), f"{model_dir}/ppo_{variant_name}_final.pt")

    # Save training log
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/{variant_name}_log.json", "w") as f:
        json.dump(log, f)

    print(f"Best win rate: {best_win_rate:.1f}%")
    return model


def evaluate_variant(variant_name, n_games=500, use_eliminated=True,
                     model_dir="models", result_dir="results"):
    cfg = REWARD_CONFIGS[variant_name]
    env = WordleWrapperExperiment(cfg, use_eliminated=use_eliminated)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model_path = f"{model_dir}/ppo_{variant_name}_best.pt"
    if not os.path.exists(model_path):
        model_path = f"{model_dir}/ppo_{variant_name}_final.pt"

    model = ActorCritic(env.state_dim, env.word_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    wins = 0
    guess_counts = []

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
        if reward > 0:
            wins += 1
            guess_counts.append(rounds)
        else:
            guess_counts.append(7)  # 7 = failed

    result = {
        "variant": variant_name,
        "label": cfg["label"],
        "win_rate": wins / n_games * 100,
        "avg_guesses": np.mean([g for g in guess_counts if g <= 6]),
        "guess_distribution": guess_counts,
        "n_games": n_games,
    }

    with open(f"{result_dir}/{variant_name}_eval.json", "w") as f:
        json.dump(result, f)

    print(f"{cfg['label']}: Win={result['win_rate']:.1f}%, "
          f"Avg guesses={result['avg_guesses']:.2f}")
    return result


# ── Plotting ─────────────────────────────────────────────────

def generate_plots(variants=None, result_dir="results"):
    if variants is None:
        variants = list(REWARD_CONFIGS.keys())

    os.makedirs(result_dir, exist_ok=True)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    # 1. Reward convergence
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, v in enumerate(variants):
        path = f"{result_dir}/{v}_log.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            log = json.load(f)
        ax.plot(log["episodes"], log["avg_reward"],
                label=REWARD_CONFIGS[v]["label"], color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Reward (per episode)")
    ax.set_title("PPO Reward Convergence Across Reward Shaping Variants")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{result_dir}/reward_convergence.png", dpi=150)
    print(f"Saved {result_dir}/reward_convergence.png")
    plt.close(fig)

    # 2. Win rate over training
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, v in enumerate(variants):
        path = f"{result_dir}/{v}_log.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            log = json.load(f)
        ax.plot(log["episodes"], log["win_rate"],
                label=REWARD_CONFIGS[v]["label"], color=colors[i % len(colors)], alpha=0.85)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("PPO Win Rate During Training Across Reward Shaping Variants")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{result_dir}/win_rate_training.png", dpi=150)
    print(f"Saved {result_dir}/win_rate_training.png")
    plt.close(fig)

    # 3. Guess distribution histogram (from evaluation)
    eval_data = {}
    for v in variants:
        path = f"{result_dir}/{v}_eval.json"
        if os.path.exists(path):
            with open(path) as f:
                eval_data[v] = json.load(f)

    if eval_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.arange(1, 9) - 0.5  # 1-7 (7=fail)
        labels_x = ["1", "2", "3", "4", "5", "6", "Fail"]
        width = 0.8 / len(eval_data)

        for i, (v, data) in enumerate(eval_data.items()):
            counts, _ = np.histogram(data["guess_distribution"], bins=bins)
            pcts = counts / len(data["guess_distribution"]) * 100
            positions = np.arange(len(labels_x)) + i * width
            ax.bar(positions, pcts, width, label=data["label"],
                   color=colors[i % len(colors)], alpha=0.85)

        ax.set_xlabel("Number of Guesses")
        ax.set_ylabel("Percentage of Games (%)")
        ax.set_title("PPO Guess Distribution Across Reward Shaping Variants")
        ax.set_xticks(np.arange(len(labels_x)) + width * (len(eval_data) - 1) / 2)
        ax.set_xticklabels(labels_x)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(f"{result_dir}/guess_histogram.png", dpi=150)
        print(f"Saved {result_dir}/guess_histogram.png")
        plt.close(fig)

        # 4. Final win rate bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [eval_data[v]["label"] for v in eval_data]
        win_rates = [eval_data[v]["win_rate"] for v in eval_data]
        bars = ax.bar(range(len(names)), win_rates,
                      color=[colors[i % len(colors)] for i in range(len(names))],
                      alpha=0.85)
        for bar, wr in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{wr:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("PPO Final Win Rate by Reward Shaping Variant")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(f"{result_dir}/win_rate_comparison.png", dpi=150)
        print(f"Saved {result_dir}/win_rate_comparison.png")
        plt.close(fig)


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward Function Ablation")
    parser.add_argument("mode", choices=["train", "eval", "plot", "all"])
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Which variants to run (default: all)")
    parser.add_argument("--no-eliminated", action="store_true",
                        help="Remove eliminated vector from state encoding")
    args = parser.parse_args()

    variants = args.variants or list(REWARD_CONFIGS.keys())
    use_elim = not args.no_eliminated
    model_dir = "models" if use_elim else "models_no_elim"
    result_dir = "results" if use_elim else "results_no_elim"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    if args.mode in ("train", "all"):
        for v in variants:
            train_variant(v, n_episodes=args.episodes,
                          use_eliminated=use_elim,
                          model_dir=model_dir, result_dir=result_dir)

    if args.mode in ("eval", "all"):
        for v in variants:
            evaluate_variant(v, n_games=args.eval_games,
                             use_eliminated=use_elim,
                             model_dir=model_dir, result_dir=result_dir)

    if args.mode in ("plot", "all"):
        generate_plots(variants, result_dir=result_dir)
