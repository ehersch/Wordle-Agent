# Wordle-Agent

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

## Wordle Environment

https://pypi.org/project/gym-wordle/
