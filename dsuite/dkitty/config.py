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

"""Shared configuration dictionaries for DKitty tasks."""

import numpy as np

from dsuite.components.robot.dynamixel_utils import CalibrationMap
from dsuite.utils.config_utils import merge_configs

# Convenience constants.
PI = np.pi

# Base configuration for a DKitty robot in simulation.
DKITTY_SIM_CONFIG = {
    'class_path': 'dsuite.components.robot:RobotComponent',
    'groups': {
        'root': {
            'qpos_indices': range(6),
            'qpos_range': [
                (-10.0, 10.0),  # Torso Tx
                (-10.0, 10.0),  # Torso Ty
                (0.0, 10.0),  # Torso Tz
                (-PI / 2, PI / 2),  # Torso Rx
                (-PI / 2, PI / 2),  # Torso Ry
                (-PI / 2, PI / 2),  # Torso Rz
            ],
            'sim_observation_noise': 0.002,  # 2cm/10m
        },
        'dkitty': {
            'actuator_indices': range(12),
            'qpos_indices': range(6, 18),
            'qpos_range': [(-0.5, 0.35), (0.0, PI / 2), (-2.0, 0.0)] * 4,
            'qvel_range': [(-PI, PI)] * 12,
            'sim_observation_noise': 0.05,
        },
    }
}

# Base configuration for a DKitty robot in hardware.
DKITTY_HARDWARE_CONFIG = merge_configs(
    DKITTY_SIM_CONFIG,
    {
        'class_path': 'dsuite.components.robot:DynamixelRobotComponent',
        'groups': {
            'dkitty': {
                'motor_ids': [10, 11, 12, 40, 41, 42, 20, 21, 22, 30, 31, 32],
            },
        }
    },
)

# Base configuration for tracking in simulation.
TRACKER_SIM_CONFIG = {
    'class_path': 'dsuite.components.tracking:TrackerComponent',
    'groups': {
        'target': {
            'site_name': 'target',
        },
        'heading': {
            'site_name': 'heading',
        },
        'torso': {
            'site_name': 'torso',
        },
    }
}

# Base configuration for tracking in hardware.
# NOTE: The `device_identifier` for each group is configured at runtime.
TRACKER_HARDWARE_CONFIG = merge_configs(
    TRACKER_SIM_CONFIG,
    {
        'class_path': 'dsuite.components.tracking:VrTrackerComponent',
        'groups': {
            'target': {
                'mimic_in_sim': True,
                'mimic_ignore_z_axis': True,
                'mimic_ignore_rotation': True,
            },
            'heading': {
                'mimic_in_sim': True,
                'mimic_ignore_z_axis': True,
                'mimic_ignore_rotation': True,
            },
            'torso': {
                'is_origin': True,
            },
        },
    },
)


# Mapping of motor ID to (scale, offset).
DEFAULT_DKITTY_CALIBRATION_MAP = CalibrationMap({
    # Front right leg.
    10: (1, -3. * PI / 2),
    11: (-1, PI),
    12: (-1, PI),
    # Front left leg.
    20: (-1, PI / 2),
    21: (1, -PI),
    22: (1, -PI),
    # Back left leg.
    30: (1, -3. * PI / 2),
    31: (1, -PI),
    32: (1, -PI),
    # Back right leg.
    40: (-1, PI / 2),
    41: (-1, PI),
    42: (-1, PI),
})
