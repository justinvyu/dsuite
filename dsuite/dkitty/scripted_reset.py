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

"""Hardware reset functions for the D'Kitty."""

from typing import Any, Dict

import numpy as np

from dsuite.components.robot import DynamixelRobotComponent, RobotState

BASEMAX = .8
MIDMAX = 2.4
FOOTMAX = 2.5

# Common parameters for all `set_state` commands.
SET_PARAMS = dict(
    error_tol=5 * np.pi / 180,  # 5 degrees
    last_diff_tol=5 * np.pi / 180,  # 5 degrees
)


def add_groups_for_reset(groups_dict: Dict[str, Any]):
    """Defines groups required to perform the reset."""
    groups_dict['base'] = dict(motor_ids=[10, 20, 30, 40])
    groups_dict['middle'] = dict(motor_ids=[11, 21, 31, 41])
    groups_dict['feet'] = dict(motor_ids=[12, 22, 32, 42])
    groups_dict['front'] = dict(motor_ids=[11, 12, 21, 22])
    groups_dict['back'] = dict(motor_ids=[31, 32, 41, 42])


def reset_standup(robot: DynamixelRobotComponent):
    """Resets the D'Kitty to a standing position.

    Detects the current configuration of the D'Kitty and decides whether which
    reset procedure to perform.

    A full reset takes about 15s.
    """
    # Extend all arms completely, then let gravity force joints down; this tells
    # the current orientation of the dkitty.

    robot.set_state({
        'base': RobotState(qpos=np.full(4, -.2)),
        'feet': RobotState(qpos=np.zeros(4)),
        'middle': RobotState(qpos=np.zeros(4))
    }, **SET_PARAMS)
    robot.set_motors_engaged('dkitty', engaged=False)
    robot.set_motors_engaged('dkitty', engaged=True)
    # Detect whether the gravity check affected the middle or base joints which
    # tells the orientation.
    # If two elbows bend, it's on its rear or face.
    # If two base joints bend inward it's on its side.
    # If all four base joints bend then it's on it's back.
    middle_state, base_state = robot.get_state(['middle', 'base'])
    midavg = np.mean(np.abs(middle_state.qpos))
    baseavg = np.mean(np.abs(base_state.qpos))
    if midavg < baseavg:
        if np.min(robot.get_state('base').qpos) < -BASEMAX:
            back_recover(robot)
        else:
            side_recover(robot)
    else:
        face_recover(robot)


def side_recover(robot: DynamixelRobotComponent):
    """Recovers the D'Kitty from a side or laid-down position."""
    # Put all base joints at the same max angle
    basepos = robot.get_state('base').qpos
    cbstate = np.array([BASEMAX, -BASEMAX, -BASEMAX, BASEMAX])
    if basepos[1] + basepos[2] < basepos[0] + basepos[3]:
        cbstate *= -1
    robot.set_state({'base': RobotState(qpos=cbstate)}, **SET_PARAMS)

    # Fully fold legs causing the dkitty to tip onto its feet, doing this and
    # the previous step separately decreases the possibility of binding.
    robot.set_state({
        'feet': RobotState(qpos=np.full(4, -FOOTMAX)),
        'middle': RobotState(qpos=np.full(4, MIDMAX))
    }, **SET_PARAMS)

    # Pivot the legs so that the kitty is now in a pouncing stance.
    robot.set_state({'base': RobotState(qpos=np.zeros(4))}, **SET_PARAMS)

    straighten_legs(robot)


def face_recover(robot: DynamixelRobotComponent):
    """Recovers the D'Kitty from a face-down or face-up position."""
    midpos = np.abs(robot.get_state('middle').qpos)
    reversestand = False
    if midpos[0] + midpos[1] < midpos[2] + midpos[3]:
        robot.set_state({
            'base': RobotState(qpos=np.array([-3.14, -3.14, 0, 0])),
            'middle':
                RobotState(qpos=np.array([MIDMAX, MIDMAX, MIDMAX, MIDMAX])),
            'feet': RobotState(
                qpos=np.array([-FOOTMAX, -FOOTMAX, -FOOTMAX, -FOOTMAX]))
        }, **SET_PARAMS)
    else:
        robot.set_state({
            'base': RobotState(qpos=np.array([0, 0, -3.14, -3.14])),
            'middle':
                RobotState(qpos=np.array([-MIDMAX, -MIDMAX, -MIDMAX, -MIDMAX])),
            'feet':
                RobotState(qpos=np.array([FOOTMAX, FOOTMAX, FOOTMAX, FOOTMAX]))
        }, **SET_PARAMS)
        reversestand = True

    # Wiggle back and forth to get the feet fully underneath the dkitty.
    base_position = np.array([BASEMAX, -BASEMAX, -BASEMAX, BASEMAX])
    robot.set_state({'base': RobotState(qpos=base_position)}, **SET_PARAMS)
    # Delay the intermediate wiggling so that the dkitty doesn't flip over.
    robot.set_state({'base': RobotState(qpos=-base_position)},
                    **SET_PARAMS,
                    initial_sleep=1.5)
    robot.set_state({'base': RobotState(qpos=np.zeros(4))}, **SET_PARAMS)

    straighten_legs(robot, reversestand)


def back_recover(robot: DynamixelRobotComponent):
    """Recovers the D'Kitty from a flipped over position."""
    # Fixed legs in straightened position and put all base joints at same max
    # angle.
    robot.set_state({
        'base':
            RobotState(qpos=np.array([-BASEMAX, BASEMAX, BASEMAX, -BASEMAX])),
        'middle': RobotState(qpos=np.zeros(4)),
        'feet': RobotState(qpos=np.zeros(4))
    }, **SET_PARAMS)

    # Using the legs to shift its CG, it now sets all of the base joints to
    # zero, raising the kitty onto its side.
    robot.set_state({'base': RobotState(qpos=np.zeros(4))}, **SET_PARAMS)

    # Then the legs are let fall, and the dkitty performs a side recover
    robot.set_motors_engaged('dkitty', engaged=False)
    robot.set_motors_engaged('dkitty', engaged=True)

    side_recover(robot)


def straighten_legs(robot: DynamixelRobotComponent, reverse: bool = False):
    """Straightens out the legs of the D'Kitty."""
    front_state, back_state = robot.get_state(['front', 'back'])
    frontpos = front_state.qpos
    backpos = back_state.qpos

    states = [
        # Actuate the front foot and knee joints halfway.
        dict(front=RobotState(qpos=frontpos / 2)),
        # Actuate the back foot and knee joints halfway.
        dict(back=RobotState(qpos=backpos / 2)),
        # Straighten the front legs.
        dict(front=RobotState(qpos=np.zeros(4))),
        # Straighten the back legs reaching a full standing position.
        dict(back=RobotState(qpos=np.zeros(4))),
    ]
    # If doing a reverse-standup, perform the back leg motions first.
    if reverse:
        states[0], states[1] = states[1], states[0]
        states[2], states[3] = states[3], states[2]

    for state in states:
        robot.set_state(state, **SET_PARAMS)
