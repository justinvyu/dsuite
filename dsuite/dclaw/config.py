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

"""Shared configuration dictionaries for DClaw tasks."""

import numpy as np

from dsuite.controllers.robot.dynamixel_utils import CalibrationMap
from dsuite.utils.config_utils import merge_configs

# Convenience constants.
PI = np.pi

# Base configuration for a DClaw robot in simulation.
DCLAW_SIM_CONFIG = {
    'class_path': 'dsuite.controllers.robot:RobotController',
    'groups': {
        'dclaw': {
            'qpos_indices': range(9),
            'qpos_range': [
                (-PI / 4, PI / 4),  # 45 degrees for top servos.
                (-PI / 3, PI / 3),  # 60 degrees for middle servos.
                (-PI / 2, PI / 2),  # 90 degrees for bottom servos.
            ] * 3,
            'qvel_range': [(-2 * PI / 3, 2 * PI / 3)] * 9,
            'sim_observation_noise': 0.05,
        },
    }
}

# Base partial configuration for the object in simulation.
_OBJECT_SIM_CONFIG = {
    'groups': {
        'object': {
            'qpos_indices': [-1],  # The object is the last qpos.
            'qpos_range': [(-PI, PI)],
            'sim_observation_noise': 0.05,
        },
        'guide': {},  # The guide group is a no-op in simulation.
    }
}

# Base partial configuration for the object in simulation.
_FREE_OBJECT_SIM_CONFIG = {
    'groups': {
        'object': {
            # 'qpos_indices': [-1],  # The object is the last qpos.
            # 'qpos_range': [(-np.pi, np.pi)],

            'qpos_indices': range(-6, 0),  # The object is the last qpos.
            'qpos_range': [
                (-0.15, 0.15),     # restrict x
                (-0.15, 0.15),     # restrict y
                (0, 0.25),          # unrestricted z
                (-np.pi, np.pi), # unrestricted object orientation
                (-np.pi, np.pi),
                (-np.pi, np.pi)],
            'sim_observation_noise': 0,
        },
        'guide': {},  # The guide group is a no-op in simulation.
    }
}

# Base configuration for a DClaw robot in hardware.
DCLAW_HARDWARE_CONFIG = merge_configs(
    DCLAW_SIM_CONFIG,
    {
        'class_path': 'dsuite.controllers.robot:DynamixelRobotController',
        'groups': {
            'dclaw': {
                'motor_ids': [10, 11, 12, 20, 21, 22, 30, 31, 32],
            },
        }
    },
)

# Base configuration for the object on hardware.
_OBJECT_HARDWARE_CONFIG = merge_configs(
    _OBJECT_SIM_CONFIG,
    {'groups': {
        'object': {
            'motor_ids': [50],
        },
    }},
)

# Configuration for a DClaw with an object in simulation.
DCLAW_OBJECT_SIM_CONFIG = merge_configs(DCLAW_SIM_CONFIG, _OBJECT_SIM_CONFIG)

# Configuration for a DClaw with a free object in simulation.
DCLAW_FREE_OBJECT_SIM_CONFIG = merge_configs(DCLAW_SIM_CONFIG, _FREE_OBJECT_SIM_CONFIG)

# Configuration for a DClaw with an object in hardware.
DCLAW_OBJECT_HARDWARE_CONFIG = merge_configs(
    DCLAW_HARDWARE_CONFIG,
    _OBJECT_HARDWARE_CONFIG,
    {
        'groups': {
            # Make a group that we disable in hardware during reset.
            'disable_in_reset': {
                'motor_ids': [10, 12, 20, 22, 30, 32, 50],
            },
        }
    },
)

# Configuration for a DClaw with an object and guide motor in hardware.
# NOTE: This motor is optional and is used to show the goal position.
DCLAW_OBJECT_GUIDE_HARDWARE_CONFIG = merge_configs(
    DCLAW_OBJECT_HARDWARE_CONFIG,
    {
        'groups': {
            'guide': {
                'motor_ids': [60],
            }
        },
    },
)

# Mapping of motor ID to (scale, offset).
DEFAULT_DCLAW_CALIBRATION_MAP = CalibrationMap({
    # Finger 1
    10: (1, -PI / 2),
    11: (1, -PI),
    12: (1, -PI),
    # Finger 2
    20: (1, -PI / 2),
    21: (1, -PI),
    22: (1, -PI),
    # Finger 3
    30: (1, -PI / 2),
    31: (1, -PI),
    32: (1, -PI),
    # Object
    50: (1, -PI),
    # Guide
    60: (1, -PI),
})
