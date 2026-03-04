import gym
import gym_wordle

env = gym.make('Wordle-v0')

n_games = 200

wins = 0
rewards = []
lens = []
for _ in range(n_games):
    obs = env.reset()
    done = False
    reward = 0
    while not done:
        while True:
            # make a random guess
            act = env.action_space.sample()

            # take a step
            raw, _, done, info = env.step(act)

            # Match env-side per-character shaping from latest guess flags.
            round_idx = env.unwrapped.round - 1
            flags = raw[round_idx][5:10]
            right_pos = getattr(env.unwrapped, "right_pos", 1)
            wrong_pos = getattr(env.unwrapped, "wrong_pos", 2)
            n_green = int((flags == right_pos).sum())
            n_yellow = int((flags == wrong_pos).sum())
            shaped = 0.25 * n_green + 0.10 * n_yellow - 0.05
            reward += shaped
            break
    rewards.append(reward)
    rounds = env.unwrapped.round
    lens.append(rounds)

avg_rew = sum(rewards) / n_games
avg_len = sum(lens) / n_games

print("Average reward: " + str(avg_rew))
print("Average len: " + str(avg_len))