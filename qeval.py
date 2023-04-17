#!/usr/bin/env python3

import argparse
import gymnasium as gym
import numpy as np
import re
from pprint import pprint


_VERSION = 2


class QQE():
    """Q-Reinforcement Learning
    Q-table
    Eval
    """

    def __init__(
            self, *,
            quiet,
            qtable_file,
            states_cover,
            max_steps=None,
            environment=None,
        ):
        self.quiet = quiet
        self.qtable_file = qtable_file
        self.states_cover = states_cover
        self.max_steps = max_steps if max_steps else 100

        self.environment = environment if environment else "Taxi-v3"

        if self.environment == "Taxi-v3":
            self.n_possible_states = 300
        else:
            self.n_possible_states = 1_000
        self.n_states = int(self.n_possible_states * self.states_cover / 100)
        self.state_tries_limit = self.n_possible_states * 100

    def setup(self):
        self.qtable = np.loadtxt(self.qtable_file)
        self.env = gym.make(
            id=self.environment,
            max_episode_steps=self.max_steps,
            render_mode=None,
        )

    def eval(self):
        self.setup()
        self.dones_steps = 0
        self.truncates = 0
        self.dones = 0
        state_set = set()
        self.episode = 0
        can_find_state = True
        if not self.quiet:
            print("Epsodes - (total_state_tries):")
        total_state_tries = 0
        while len(state_set) < self.n_states:

            state_tries = 0
            while True:
                state, info = self.env.reset()
                if state in state_set:
                    state_tries += 1
                    total_state_tries +=1
                    if state_tries > self.state_tries_limit:
                        can_find_state = False
                        break
                    continue
                state_set.add(state)
                break
            if not can_find_state:
                break

            steps = 0
            done = truncated = False
            while not (done or truncated):
                action = np.argmax(self.qtable[state])
                state, reward, done, truncated, info = self.env.step(action)
                steps += 1
            if done:
                self.dones += 1
                self.dones_steps += steps
            else:
                self.truncates += 1
            if not self.quiet:
                print(f"\r{self.episode+1:6d} - ({total_state_tries:9d})", end='')
            self.episode += 1

        self.percent_dones = self.dones / self.episode * 100
        if self.dones:
            self.avg_steps = self.dones_steps / self.dones
        else:
            self.avg_steps = 0
        epsilon_type, train_epsodes = re.findall(f'qgye{_VERSION}_([^_]+)_\d+_\d+-(\d+).qtb', args.qtable)[0]
        if not self.quiet:
            print()
            print(f"Done: {self.percent_dones:.2f} {self.dones}; Truncated: {self.truncates}")
        if self.dones != 0:
            if self.quiet:
                print(f'"{epsilon_type}";{train_epsodes};{self.percent_dones:.2f};{self.avg_steps:.6f}')
            else:
                print(f"Average timesteps per doned episode: {self.avg_steps:.6f}")
        else:
            if self.quiet:
                print(f'"{epsilon_type}";{train_epsodes};0;0')

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
        description=f"Eval Q-Reinforcement Learning Q-table - v.{_VERSION}",
        epilog="(c) Anselmo Blanco Dominguez",
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
    )
    parser.add_argument(
        'qtable',
        help="file with saved Q-table",
    )
    parser.add_argument(
        'states_cover',
        help="percentage of states covered [1, 100]",
        type=int_limits(start=1, end=100),
    )
    parser.add_argument(
        'max_steps',
        help="maximum steps per episode [1, 10^6]",
        type=int_limits(start=1, end=10**6),
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    qqe = QQE(
        quiet=args.quiet,
        qtable_file=args.qtable,
        max_steps=args.max_steps,
        states_cover=args.states_cover,
    )
    qqe.eval()
