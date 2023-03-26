#!/usr/bin/env python3

import argparse
import gymnasium as gym
import math
import numpy as np
import os
import random
import time
from pprint import pprint
  

class QGE():
    """Q-Reinforcement Learning - Gym
    Epsilon analysis
    """

    def __init__(self, epsilon_option, num_episodes, max_steps, environment=None):
        self.qtb_dir = 'qtb'
        self.version = 'qgye1'

        self.epsilon_option = epsilon_option
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.environment = "Taxi-v3" if environment is None else environment

        print("Q-rl Gym Epsilon analysis")
        print("epsilon_option:", epsilon_option)
        print("num_episodes:", num_episodes)
        print("max_steps:", max_steps)
        print()

    def setup(self, alpha=None, gamma=None):

        # Hyperparameters

        # Learning rate
        self.alpha = 0.1 if alpha is None else alpha
        # 0.1 means
        # - 10% of the value learned in the step
        # - 90% of the previous value   

        # fator de desconto
        self.gamma = 0.6 if gamma is None else gamma
        # 0.6 significa
        # - the reward for the current action is added to 60%
        #   of the maximum reward from the next state onwards

        print("Initialize the environment")
        print("alpha:", alpha)
        print("gamma:", gamma)

        self.env = gym.make(
            self.environment,
            max_episode_steps=self.max_steps,
            render_mode=None,
        )
        print("observation space:", self.env.observation_space.n)
        print("action space:", self.env.action_space.n)

        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        self.q_table_size = self.env.observation_space.n * self.env.action_space.n
        print("q_table_size:", self.q_table_size)

        print("zeros:", np.count_nonzero(self.q_table==0))
        print()

    def epsilon_value(self):
        # explore X exploit
        # 0.1 means
        # - 10% probability of choosing an action at random
        # - 90% of choosing the action with the best reward,
        #   according to what is known so far
        if self.epsilon_option == 'a':
            return 0.1
        elif self.epsilon_option == 'b':
            return 0.9
        elif self.epsilon_option == 'c':
            return (np.count_nonzero(self.q_table == 0) / self.q_table_size) * 0.9 + 0.1
        elif self.epsilon_option == 'd':
            return 0.05 + (1 - 0.05) * math.e ** (-self.episode / 6000)

    def run_episode(self):
        state, info = self.env.reset()
        epoch, penalties, reward = 0, 0, 0

        done = False
        truncated = False
        while not (done or truncated):
            self.epsilon = self.epsilon_value()

            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(self.q_table[state])  # Exploit learned values

            next_state, reward, done, truncated, info = self.env.step(action)

            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state])
            new_value = (
                (1 - self.alpha) * old_value +
                self.alpha * (reward + self.gamma * next_max)
            )
            self.q_table[state, action] = new_value

            state = next_state
            epoch += 1

            if reward == -10:
                penalties += 1
            if done:
                self.dones += 1
            elif truncated:
                self.truncs += 1
            print(
                f"\repisode {self.episode:6d}; epoch {epoch:6d}; "
                f"dones {self.dones:6d}; truncs {self.truncs:6d}; "
                f"zeros in q_table {np.count_nonzero(self.q_table==0):6d}; "
                , end=''
            )
            # time.sleep(0.005)

    def txt_filename(self):
        txt_dir = os.path.join(
            self.qtb_dir,
            self.version,
        )
        if not os.path.isdir(txt_dir):
            os.makedirs(txt_dir)
        return os.path.join(
            txt_dir,
            f"{self.version}_{self.epsilon_option}"
            f"_{self.num_episodes}_{self.max_steps}.qtb"
        )

    def save_qtable(self):
        txt_filename = self.txt_filename()
        print("Save qtable to", txt_filename)
        np.savetxt(txt_filename, self.q_table)
        print()

    def train(self):
        self.setup()

        print(f"Running {self.num_episodes} episodes")

        self.dones = 0
        self.truncs = 0
        for self.episode in range(self.num_episodes):
            self.run_episode()
        print("\n")

        self.save_qtable()

        print("Training finished")


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
        description="Train Q-learning Reinforcement Learning",
        epilog="(c) Anselmo Blanco Dominguez",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        'epsilon_option',
        help=(
            "Epsilon hyperparameter use option\n"
            "  a: epsilon=0.1\n"
            "  b: epsilon=0.9\n"
            "  c: epsilon=(zeros em q_table) / q_table size) * 0.9 + 0.1\n"
            "  d: epsilon=0.05 + (1 - 0.05) * e ** (-episode / 6000)\n"
        ),
        choices=['a', 'b', 'c', 'd']
    )
    parser.add_argument(
        'num_episodes',
        help="number of episodes [1, 10^9]",
        type=int_limits(start=1, end=10**9),
    )
    parser.add_argument(
        'max_steps',
        help="maximum steps per episode [1, 10^6]",
        type=int_limits(start=1, end=10**6),
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    q = QGE(args.epsilon_option, args.num_episodes, args.max_steps)
    q.train()
