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

"""Helper functions to load environments in scripts."""

import argparse
import os
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

from gym.envs import registration as gym_reg


def parse_env_args(
        arg_parser: Optional[argparse.ArgumentParser] = None,
        default_env_name: Optional[str] = None,
) -> Tuple[str, Dict, argparse.Namespace]:
    """Parses the given arguments to get an environment ID and parameters.

    Args:
        arg_parser: An existing argument parser to add arguments to.

    Returns:
        env_name: The name of the environment.
        env_params: A dictionary of environment parameters that can be passed
            to the constructor of the environment class, or passed via
            `dsuite.set_env_params`.
        args: The Namespace object parsed by the ArgumentParser.
    """
    if arg_parser is None:
        arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-e',
        '--env_name',
        required=(default_env_name is None),
        default=default_env_name,
        help='The environment to load.')
    arg_parser.add_argument('-d', '--device', help='The device to connect to.')
    arg_parser.add_argument(
        '--param',
        action='append',
        help=('A "key=value" pair to pass as an environment parameter. This '
              'be repeated, e.g. --param key1=val1 --param key2=val2'))
    arg_parser.add_argument(
        '--info', action='store_true', help='Turns on info logging.')
    arg_parser.add_argument(
        '--debug', action='store_true', help='Turns on debug logging.')
    args = arg_parser.parse_args()

    # Ensure the environment ID is valid.
    env_name = args.env_name
    if env_name not in gym_reg.registry.env_specs:
        raise ValueError('Unregistered environment ID: {}'.format(env_name))

    # Ensure the device exists, if given.
    device_path = args.device
    if device_path and not os.path.exists(device_path):
        raise ValueError('Device does not exist: {}'.format(device_path))

    # Parse environment params into a dictionary.
    env_params = {}
    if args.param:
        env_params = parse_env_params(args.param)

    if device_path:
        env_params['device_path'] = device_path

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.info:
        logging.basicConfig(level=logging.INFO)

    return env_name, env_params, args


def parse_env_params(user_entries: Sequence[str]) -> Dict[str, Any]:
    """Parses a list of `key=value` strings as a dictionary."""

    def is_value_convertable(v, convert_type) -> bool:
        try:
            convert_type(v)
        except ValueError:
            return False
        return True

    env_params = {}
    for user_text in user_entries:
        components = user_text.split('=')
        if len(components) != 2:
            raise ValueError('Key-values must be specified as `key=value`')
        value = components[1]
        if is_value_convertable(value, int):
            value = int(value)
        elif is_value_convertable(value, float):
            value = float(value)
        env_params[components[0]] = value
    return env_params
