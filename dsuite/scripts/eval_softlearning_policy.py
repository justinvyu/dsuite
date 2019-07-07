# Copyright 2019 The D'Suite Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a softlearning policy.

Example usage:
python -m dsuite.scripts.eval_softlearning_policy --device /dev/ttyUSB0
"""

import argparse
import collections
import csv
import os
import pickle
from typing import Any, Dict, Optional

import gym
import numpy as np

from softlearning.environments.adapters import gym_adapter
from softlearning.samplers import rollout

import dsuite
from dsuite.scripts.utils import parse_env_args

gym_adapter.DEFAULT_OBSERVATION_KEY = 'obs'

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

DEFAULT_ENV_NAME = 'DClawTurnRandom-v0'
DEFAULT_POLICY_PATH = os.path.join(SCRIPT_DIR,
                                   'data/DClawTurnRandom-v0-policy.pkl')
DEFAULT_EPISODE_COUNT = 10
DEFAULT_EPISODE_LEN = 160


class CsvLogger:
    """Logs data to a CSV file."""

    def __init__(self, path: str):
        self._file = open(path, 'w')
        self._writer = None

    def write(self, data: Dict[str, Any]):
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(data.keys()))
            self._writer.writeheader()
        self._writer.writerow(data)
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def do_rollouts(env,
                policy,
                logger: CsvLogger,
                num_episodes: int,
                episode_length: int,
                render_mode: Optional[str] = None):
    render_kwargs = {}
    if render_mode:
        render_kwargs['mode'] = render_mode
    total_timesteps = 0
    for i in range(num_episodes):
        path = rollout(
            env,
            policy,
            path_length=episode_length,
            render_kwargs=render_kwargs)
        total_timesteps += len(path['rewards'])
        reward = path['rewards'].sum()

        data = collections.OrderedDict((
            ('episode', i),
            ('total_timesteps', total_timesteps),
            ('reward', reward),
        ))

        for info_key, info_values in path.get('infos', {}).items():
            data[info_key + '-first-mean'] = np.mean(info_values[0])
            data[info_key + '-last-mean'] = np.mean(info_values[-1])
            data[info_key + '-mean-mean'] = np.mean(info_values)

        logger.write(data)
        print('[{}] Episode : {}'.format(i, reward))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--output',
        default='output',
        help='The output directory to save evaluation data to.')
    parser.add_argument(
        '-p',
        '--policy',
        default=DEFAULT_POLICY_PATH,
        help='The path to the pickled softlearning policy to load.')
    parser.add_argument(
        '-n',
        '--num_episodes',
        default=DEFAULT_EPISODE_COUNT,
        type=int,
        help='The number of episodes the evaluate.')
    parser.add_argument(
        '-l',
        '--episode_length',
        default=DEFAULT_EPISODE_LEN,
        type=int,
        help='The number of steps in each episode.')
    parser.add_argument(
        '-r',
        '--render',
        nargs='?',
        const='human',
        default=None,
        choices=['human', 'rgb_array'],
        help='The render mode for the policy.')
    env_name, env_params, args = parse_env_args(
        parser, default_env_name=DEFAULT_ENV_NAME)

    # Load the environment.
    env = gym.make(env_name, **env_params)
    env = gym_adapter.GymAdapter(env=env, domain=None, task=None)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load the policy.
    with open(args.policy, 'rb') as f:
        policy_data = pickle.load(f)
        policy = policy_data['policy']
        policy.set_weights(policy_data['weights'])

    with CsvLogger(os.path.join(args.output, 'results.csv')) as logger:
        with policy.set_deterministic(True):
            do_rollouts(
                env,
                policy,
                logger,
                num_episodes=args.num_episodes,
                episode_length=args.episode_length,
                render_mode=args.render)


if __name__ == '__main__':
    main()
