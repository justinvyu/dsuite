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

from dsuite.controllers.robot import DynamixelRobotController, RobotState


def add_groups_for_reset(groups_dict: Dict[str, Any]):
    """Defines groups required to perform the reset."""
    groups_dict['base'] = dict(motor_ids=[10, 20, 30, 40])
    groups_dict['middle'] = dict(motor_ids=[11, 21, 31, 41])
    groups_dict['feet'] = dict(motor_ids=[12, 22, 32, 42])
    groups_dict['front'] = dict(motor_ids=[11, 12, 21, 22])
    groups_dict['back'] = dict(motor_ids=[31, 32, 41, 42])


def reset_standup(robot: DynamixelRobotController):
    """Resets the D'Kitty to a standing position."""
    basemax = .8
    midmax = 2.4
    footmax = 2.5

    # Put all base joints at the same angle.
    baseavg = np.sum(robot.get_state('base').qpos) / 4
    cbstate = np.array([basemax, -basemax, -basemax, basemax])
    if baseavg < 0:
        cbstate *= -1
    robot.set_state({'base': RobotState(qpos=cbstate)})

    # Fold in all four arms to max fold, this causes the kitty to tilt onto its
    # feet from a side position.
    midfoldstate = np.full(4, midmax)
    footfoldstate = np.full(4, -footmax)
    if baseavg < 0:
        midfoldstate *= -1
        footfoldstate *= -1

    robot.set_state({
        'feet': RobotState(qpos=footfoldstate),
        'middle': RobotState(qpos=midfoldstate),
    })

    # Pivot the legs so that the kitty is now in a "ready to pounce" stance.
    robot.set_state({'base': RobotState(qpos=np.zeros(4))})
    front_state, back_state = robot.get_state(['front', 'back'])
    frontpos = front_state.qpos
    backpos = back_state.qpos

    # Actuate the front foot and knee joints halfway.
    robot.set_state({'front': RobotState(qpos=frontpos / 2)})

    # Actuate the back foot and knee joints halfway.
    robot.set_state({'back': RobotState(qpos=backpos / 2)})

    # Straighten the front legs.
    robot.set_state({'front': RobotState(qpos=np.zeros(4))})

    # Straighten the back legs reaching a full standing position.
    robot.set_state({'back': RobotState(qpos=np.zeros(4))})
