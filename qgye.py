#!/usr/bin/env python3

import argparse
import csv
import gymnasium as gym
import math
import numpy as np
import os
import random
from pprint import pprint


class CSVWriter():

    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, 'w', encoding='utf8')
        self.writer = csv.writer(self.fp, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')

    def close(self):
        self.fp.close()

    def write(self, *data):
        self.writer.writerow(data)


class QGE():
    """Q-Reinforcement Learning
    Gymnasium
    Epsilon analysis
    """

    def __init__(
            self, *,
            epsilon_option,
            num_episodes,
            max_steps,
            quiet=False,
            verbose=0,
            environment=None,
            qtable_saves=[],
        ):
        self.qtb_dir = 'qtb'
        self.version = 'qgye1'

        self.epsilon_option = epsilon_option
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.quiet = quiet
        self.verbose = verbose
        self.environment = "Taxi-v3" if environment is None else environment
        self.qtable_saves = qtable_saves

        self.prt("Q-RL Gymnasium Epsilon analysis")
        self.prtv("  epsilon_option:", epsilon_option)
        self.prtv("  num_episodes:", num_episodes)
        self.prtv("  max_steps:", max_steps)

        self._output_filename = None
        self._qtb_filename = None
        self._csv_filename = None

    @property
    def epsilon(self):
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

    @property
    def output_filename(self):
        if not self._output_filename:
            self._output_filename = self.filename()
        return self._output_filename

    @property
    def qtb_filename(self):
        if not self._qtb_filename:
            self._qtb_filename = f"{self.output_filename}.qtb"
        return self._qtb_filename

    @property
    def csv_filename(self):
        if not self._csv_filename:
            self._csv_filename = f"{self.output_filename}.csv"
        return self._csv_filename

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
        self._done = value
        if value:
            self.dones += 1

    @property
    def truncated(self):
        return self._truncated

    @truncated.setter
    def truncated(self, value):
        self._truncated = value
        if value:
            self.truncs += 1

    def level_print(self, level, *args, **kwargs):
        if not self.quiet:
            if level <= self.verbose:
                print(*args, **kwargs)

    def prt(self, *args, **kwargs):
            self.level_print(0, *args, **kwargs)

    def prtv(self, *args, **kwargs):
            self.level_print(1, *args, **kwargs)

    def prtvv(self, verbose, *args, **kwargs):
            self.level_print(verbose, *args, **kwargs)

    def filename(self):
        txt_dir = os.path.join(
            self.qtb_dir,
            self.version,
        )
        if not os.path.isdir(txt_dir):
            os.makedirs(txt_dir)
        txt_filename = '_'.join(map(str,[
            self.version,
            self.epsilon_option,
            self.num_episodes,
            self.max_steps,
        ]))
        return os.path.join(txt_dir, txt_filename)

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

        self.prtv("Initialize the environment")
        self.prtv("  alpha:", alpha)
        self.prtv("  gamma:", gamma)

        self.env = gym.make(
            self.environment,
            max_episode_steps=self.max_steps,
            render_mode=None,
        )
        self.prtv("  observation space:", self.env.observation_space.n)
        self.prtv("  action space:", self.env.action_space.n)

        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        self.q_table_size = self.env.observation_space.n * self.env.action_space.n
        self.prtv("  q_table_size:", self.q_table_size)

        self.prtv("  zeros:", np.count_nonzero(self.q_table==0))

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore action space
        else:
            return np.argmax(self.q_table[state])  # Exploit learned values

    def update_qtable(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (
            (1 - self.alpha) * old_value +
            self.alpha * (reward + self.gamma * next_max)
        )
        self.q_table[state, action] = new_value

    def init_steps(self):
        self.step = 0
        self.done = False
        self.truncated = False

    def print_step(self):
        mantem_print_anterior = (
            self.verbose >= 3 or
            (
                self.verbose == 2 and
                self.step == 0 and
                self.episode != 0
            )
        )
        controle = "\n" if mantem_print_anterior else "\r"
        informacao = (
            f"episode {self.episode:6d}; "
            f"step {self.step:6d}; "
            f"dones {self.dones:6d}; "
            f"truncs {self.truncs:6d}; "
            f"zeros in q_table {np.count_nonzero(self.q_table==0):6d}; "
        )
        self.prt(f"{controle}  {informacao}", end='')

    def end_print_step(self):
        if self.episode == (self.num_episodes - 1):
            self.prt()

    def run_episode(self):
        state, info = self.env.reset()
        self.init_steps()
        while not (self.done or self.truncated):
            action = self.get_action(state)
            (
                next_state,
                reward,
                self.done,
                self.truncated,
                info,
            ) = self.env.step(action)
            self.update_qtable(state, action, reward, next_state)
            state = next_state
            self.print_step()
            self.step += 1
        self.train_data.write(
            self.episode,
            self.dones,
            self.truncs,
            np.count_nonzero(self.q_table==0),
        )
        self.end_print_step()

    def save_qtable(self, extra_save=None):
        extra_save = f"-{extra_save}" if extra_save else ''
        filename = f"{self.output_filename}{extra_save}.qtb"
        self.prt("Save qtable to", filename)
        np.savetxt(filename, self.q_table)

    def train(self):
        self.setup()

        self.prt(f"Training started - Running {self.num_episodes} episodes")
        self.dones = 0
        self.truncs = 0
        self.train_data = CSVWriter(self.csv_filename)
        self.train_data.write(
            'episode', 'dones', 'truncs', 'q_table_zeros')
        for self.episode in range(self.num_episodes):
            self.run_episode()
            if (self.episode+1) in self.qtable_saves:
                self.save_qtable(self.episode+1)
        self.train_data.close()
        self.prt("Training finished")

        self.save_qtable()


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


class SplitIntArg(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, list(map(int, values.split(','))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Q-Reinforcement Learning",
        epilog="(c) Anselmo Blanco Dominguez",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-q',
        '--quiet',
        action='store_true',
    )
    group.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
    )
    parser.add_argument(
        'epsilon_option',
        help=(
            "epsilon hyperparameter use option\n"
            "  a: epsilon=0.1\n"
            "  b: epsilon=0.9\n"
            "  c: epsilon=(zeros em q_table / q_table size) * 0.9 + 0.1\n"
            "  d: epsilon=0.05 + (1 - 0.05) * e ** (-episode / 6000)\n"
        ),
        choices='abcd',
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
    parser.add_argument(
        '-s',
        '--qtable_saves',
        help="comma separated list of episode number when q-table is also saved",
        default=[],
        action=SplitIntArg,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    q = QGE(
        epsilon_option=args.epsilon_option,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        quiet=args.quiet,
        verbose=args.verbose,
        qtable_saves=args.qtable_saves,
    )
    q.train()
