#!/usr/bin/env python3

import argparse
import gymnasium as gym
import numpy as np
import sys
from pprint import pprint


class QQE():
    """Q-Reinforcement Learning
    Q-table
    Eval
    """

    def __init__(
            self, *,
            qtable_file,
            num_episodes,
            max_steps=None,
            environment=None,
        ):
        self.qtable_file = qtable_file
        self.num_episodes = num_episodes
        self.environment = environment if environment else "Taxi-v3"
        self.max_steps = max_steps if max_steps else 100

    def setup(self):
        self.qtable = np.loadtxt(self.qtable_file)
        self.env = gym.make(
            self.environment,
            max_episode_steps=self.max_steps,
            render_mode=None,
        )

    def eval(self):
        self.setup()
        self.total_steps = 0
        self.total_penalties = 0
        self.total_truncates = 0
        self.total_dones = 0
        for self.episode in range(self.num_episodes):
            state, info = self.env.reset()
            steps, penalties, reward = 0, 0, 0
            done = truncated = False
            state_action = []
            while not (done or truncated):
                action = np.argmax(self.qtable[state])
                state_action.append((state, action))
                state, reward, done, truncated, info = self.env.step(action)

                if reward <= -10:
                    penalties += 1
                if done:
                    self.total_dones += 1
                if truncated:
                    # pprint(state_action)
                    self.total_truncates += 1
                steps += 1
            if done:
                self.total_penalties += penalties
                self.total_steps += steps
            print(f"\b\b\b\b\b\b{self.episode+1:6d}", end='')

        print()
        print(f"Doned: {self.total_dones}; Truncated: {self.total_truncates}")
        if self.total_dones != 0:
            print(f"Average timesteps per doned episode: {self.total_steps / self.total_dones}")
            print(f"Average penalties per doned episode: {self.total_penalties / self.total_dones}")


def int_limits(start=None, end=None):
    """Return type function to use in argparse parser.add_argument
    input: start and end (included limits)
    """
    def convert_verify(value, start=start, end=end):
        try:
            value = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Not integer value")
        if value < start or value > end:
            raise argparse.ArgumentTypeError("Integer out of bounds")
        return value
    return convert_verify


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval Q-Reinforcement Learning Q-table",
        epilog="(c) Anselmo Blanco Dominguez",
    )
    parser.add_argument(
        'qtable',
        help="file with saved Q-table",
    )
    parser.add_argument(
        'num_episodes',
        help="number of episodes [1, 10^9]",
        type=int_limits(start=1, end=10**9),
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    qqe = QQE(
        qtable_file=args.qtable,
        num_episodes=args.num_episodes,
    )
    qqe.eval()
