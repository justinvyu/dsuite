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

"""Script to test resetting robot hardware environments.

To run:
python -m dsuite.scripts.reset_hardware \
    -e DKittyWalkFixed-v0 -d /dev/ttyUSB0
"""

import argparse
import time

import gym

import dsuite
from dsuite.components.robot import DynamixelRobotComponent
from dsuite.scripts.utils import parse_env_args


def main():
    # Get command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--num_repeats',
        type=int,
        default=1,
        help='The number of resets to perform.')
    env_id, params, args = parse_env_args(parser)

    # Create the environment and get the robot component.
    dsuite.set_env_params(env_id, params)
    env = gym.make(env_id).unwrapped
    assert isinstance(env.robot, DynamixelRobotComponent)

    for i in range(args.num_repeats):
        print('Starting reset #{}'.format(i))

        # Disengage all of the motors and let the dkitty fall.
        env.robot.set_motors_engaged(None, engaged=False)

        print('Place the robot to a starting position.')
        input('Press Enter to start the reset...')

        # Start with all motors engaged.
        env.robot.set_motors_engaged(None, engaged=True)
        env.reset()

        print('Done reset! Turning off the robot in a few seconds.')
        time.sleep(2)


if __name__ == '__main__':
    main()
