import gym
import gym_wordle

env = gym.make('Wordle-v0')

n_games = 100

wins = 0
avg_len = 0
for _ in range(n_games):
    obs = env.reset()
    done = False
    while not done:
        while True:
            # make a random guess
            act = env.action_space.sample()

            # take a step
            obs, reward, done, _ = env.step(act)
            break

        env.render()
    
    rounds = env.unwrapped.round
    print(rounds)