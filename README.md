# Wordle-Agent

CS234 Final Project: Exploring efficient exploration strategies for Wordle using reinforcement learning and information theory.

## Agents

### Greedy Entropy Baseline (3Blue1Brown strategy)
Pure information-theoretic solver. Precomputes a 2315x2315 pattern matrix, then at each step picks the word that maximizes expected information gain (Shannon entropy of the pattern distribution). No learning — just computation. Achieves **100% win rate, ~3.7 avg rounds**.

```bash
python entropy_ppo.py baseline --games 2314              # full evaluation over all words
python entropy_ppo.py baseline --games 50 --verbose      # see per-round reasoning
python entropy_ppo.py baseline --first-word salet         # use a specific first word
python entropy_ppo.py validate                            # verify pattern matrix matches gym
```

### Entropy-Guided PPO
Combines the greedy entropy computation with PPO reinforcement learning. Policy logits are `beta * entropy_scores + learned_correction`, where entropy_scores are the exact expected info gain per word and the neural network learns small corrections. The agent starts near-optimal and RL can discover multi-step strategies that beat greedy entropy.

```bash
python entropy_ppo.py train --episodes 5000                            # train the RL agent
python entropy_ppo.py eval --model entropy_ppo_best.pt --games 200     # evaluate + compare to greedy baseline
python entropy_ppo.py play --model entropy_ppo_best.pt                 # watch one game with entropy reasoning
```

### Vanilla PPO
Pure RL baseline with no information-theoretic guidance. Uses 292-dim state encoding and dot-product attention over word embeddings. Demonstrates the difficulty of exploration in Wordle's large action space (~0% win rate).

```bash
python ppo.py train --episodes 50000
```

### DQN
Deep Q-Network baseline (see remote branch). Also struggles with Wordle's 2315-action space.

## Installing Dependencies

Start with

```
python -m pip install -U pip setuptools wheel
conda create -n wordle-env python=3.9
```

Then `conda activate wordle-env`

`pip install -r requirements.txt`

## Issues with Homebrew and Python

I'm running Homebrew Python:

`which python` -> `/opt/homebrew/...`

`sys.executable` -> `/opt/homebrew/...`

`gym_wordle` imported from → `/opt/homebrew/...`

So edits in `.../envs/wordle-env/.../site-packages/` will never be seen.

To actually get these edits, might need

`python -m pip install -e .`

## New way to install env

```
# downgrade build tooling so gym 0.19 installs
python -m pip install "pip<23" "setuptools==57.5.0" "wheel==0.37.1"

# install numpy first (gym + old libs behave better with numpy<2)
python -m pip install "numpy<2"

# install gym 0.19
python -m pip install "gym==0.19.0"

# now install gym-wordle + sty (no need to let resolver fight)
python -m pip install "gym-wordle==0.1.3" "sty==1.0.6"
```

Then to confirm

```
python -c "import gym, gym_wordle; import numpy as np; print('gym', gym.**version**, 'numpy', np.**version**)"
python -c "import gym_wordle.utils as u; print('gym_wordle path:', u.**file**)"
```

## Issues with Wordle Env

Line 35 in utils should use `lower()`

Line 225 in wordle hould be `state` not `states`

## The key to make sure not pointing to HomeBrew

```
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r
```

```
which python
python -c "import sys; print(sys.executable)"
python -c "import gym_wordle.utils as u; print(u.__file__)"
```

should not show `/opt/homebrew/...`

## Wordle Environment

https://pypi.org/project/gym-wordle/
