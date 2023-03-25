#!/usr/bin/env python3
# aprendendo a utilizar o Gymnasium
import gymnasium as gym
from pprint import pprint


# env = gym.make("Taxi-v3")

env = gym.make("LunarLander-v2", render_mode="human")
# pprint(env)
# pprint(env.__dict__)
# pprint(env.env.__dict__)
# pprint(env.env.env.__dict__)
# pprint(env.env.env.env.__dict__)
# raise SystemExit

observation, info = env.reset(seed=42)
# pprint(observation)
# pprint(info)
# raise SystemExit

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    # pprint(observation)
    # pprint(reward)
    # pprint(terminated)
    # pprint(truncated)
    if info:
        pprint(info)
    # raise SystemExit

    if terminated or truncated:
        observation, info = env.reset()

env.close()