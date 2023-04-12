#!/usr/bin/env python3

import gymnasium as gym
import numpy as np


env = gym.make(
    "Taxi-v3",
    max_episode_steps=1000,
    render_mode=None,
)

env = env.unwrapped # to access the inner functionalities of the class
env.state = np.array([-0.4, 0])
print(env.state)

for i in range(50):
    (
        next_state,
        reward,
        done,
        truncated,
        info,
    ) = env.step(1) # Just taking right in every step   
    print(next_state, env.state) #the next_state and env.state are same
    env.render()
