#!/usr/bin/env python3

import gymnasium as gym

seed = 42

env = gym.make("Taxi-v3")
state, info = env.reset(seed=seed)
print(state, info)

#############

import gymnasium as gym
import numpy as np
import random

seed = 42

random.seed(seed)
np.random.seed(seed)

env = gym.make("Taxi-v3")
state, info = env.reset(seed=seed)

print(state, info)
