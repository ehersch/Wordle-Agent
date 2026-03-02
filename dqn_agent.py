from dqn_model import DQN, Transition, ReplayBuffer

import math
import random
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gym
except ImportError as exc:
    raise RuntimeError(
        "`gym` is not installed. Install gym==0.19.0 in your wordle environment."
    ) from exc


def make_wordle_env():
    """Ensure Wordle env is registered before gym.make."""
    try:
        import gym_wordle  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "`gym_wordle` is not installed in this Python environment. "
            "Activate your wordle conda env and run: pip install gym-wordle==0.1.3 sty==1.0.6"
        ) from exc

    try:
        return gym.make("Wordle-v0")
    except Exception as exc:
        raise RuntimeError(
            "Failed to create `Wordle-v0`. Confirm you're running the same interpreter where gym_wordle is installed."
        ) from exc


env = make_wordle_env()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 50000
TAU = 0.01
LR = 5e-5


def build_solution_action_subset():
    """Map solution words into valid guess-space action indices."""
    action_ids = []
    for i in range(env.solution_space.n):
        solution_word = env.solution_space[i]
        guess_idx = env.action_space.index_of(solution_word)
        if guess_idx >= 0:
            action_ids.append(int(guess_idx))
    if not action_ids:
        raise RuntimeError("No valid mapped actions found from solution space.")
    return np.array(action_ids, dtype=np.int64)


ACTION_ID_MAP = build_solution_action_subset()

n_actions = len(ACTION_ID_MAP)
n_observations = 60

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayBuffer(50000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    return torch.tensor(
        [[random.randrange(n_actions)]], device=device, dtype=torch.long
    )


def plot_rewards(rewards, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float)

    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rewards_t.numpy(), label="episode return", alpha=0.6)

    if len(rewards_t) >= 50:
        means = rewards_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.plot(means.numpy(), label="50-episode mean", linewidth=2)
    plt.legend(loc="best")
    plt.pause(0.001)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states_list = [s for s in batch.next_state if s is not None]
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if non_final_next_states_list:
            non_final_next_states = torch.cat(non_final_next_states_list)
            next_state_values[non_final_mask] = (
                target_net(non_final_next_states).max(1).values
            )

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def evaluate_policy(n_eval=100):
    wins = 0
    total_return = 0.0

    for _ in range(n_eval):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        while True:
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)

            env_action = int(ACTION_ID_MAP[action.item()])
            observation, reward, done, _ = env.step(env_action)
            total_return += reward
            if done:
                if reward >= 0:
                    wins += 1
                break
            state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

    return wins / n_eval, total_return / n_eval


def main():
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 5000
    else:
        num_episodes = 3000

    episode_returns = []
    print(
        f"Using reduced action space: {n_actions} actions "
        f"(from full {env.action_space.n})"
    )

    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        ep_return = 0.0
        for _ in count():
            action = select_action(state)
            env_action = int(ACTION_ID_MAP[action.item()])
            observation, reward_val, done, _ = env.step(env_action)
            ep_return += reward_val
            reward = torch.tensor([reward_val], device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state
            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_returns.append(ep_return)
                break

        if i_episode % 25 == 0:
            plot_rewards(episode_returns)

        if i_episode % 100 == 0 and i_episode > 0:
            win_rate, eval_ret = evaluate_policy(n_eval=100)
            mean_100 = (
                float(np.mean(episode_returns[-100:]))
                if len(episode_returns) >= 100
                else float(np.mean(episode_returns))
            )
            print(
                f"ep={i_episode:4d} train_mean100={mean_100:7.3f} "
                f"eval_win={win_rate:6.2%} eval_ret={eval_ret:7.3f}"
            )

    plot_rewards(episode_returns, show_result=True)
    print("Complete")
    plt.show()


if __name__ == "__main__":
    main()
