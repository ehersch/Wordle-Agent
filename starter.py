import gym
import gym_wordle

env = gym.make("Wordle-v0")

done = False
while not done:
    action = ...  # RL magic
    state, reward, done, info = env.step(action)
