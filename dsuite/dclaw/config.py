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

from dsuite.utils.config_utils import merge_configs

# Base configuration for a DClaw robot in simulation.
DCLAW_SIM_CONFIG = {
    'class_path': 'dsuite.controllers.robot:RobotController',
    'groups': {
        'dclaw': {
            'qpos_indices': range(9),
            'qpos_range': [
                (-np.pi / 4, np.pi / 4),  # 45 degrees for top servos.
                (-np.pi / 3, np.pi / 3),  # 60 degrees for middle servos.
                (-np.pi / 2, np.pi / 2),  # 90 degrees for bottom servos.
            ] * 3,
            'qvel_range': [(-1.5, 1.5)] * 9,
            'sim_observation_noise': 0.05,
        },
    }
}

# Base partial configuration for the object in simulation.
_OBJECT_SIM_CONFIG = {
    'groups': {
        'object': {
            'qpos_indices': [-1],  # The object is the last qpos.
            'qpos_range': [(-np.pi, np.pi)],
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
                'calib_scale': [1] * 9,
                'calib_offset': [-np.pi / 2, -np.pi, -np.pi] * 3,
            },
        }
    },
)

# Base configuration for the object on hardware.
_OBJECT_HARDWARE_CONFIG = merge_configs(
    _OBJECT_SIM_CONFIG,
    {
        'groups': {
            'object': {
                'motor_ids': [50],
                'calib_scale': [1],
                'calib_offset': [-np.pi],
            },
        }
    },
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
                'calib_scale': [1] * 7,
                'calib_offset': [-np.pi / 2, -np.pi] * 3 + [-np.pi],
            },
        }
    },
)

# Configuration for a DClaw with an object and guide motor in hardware.
DCLAW_OBJECT_GUIDE_HARDWARE_CONFIG = merge_configs(
    DCLAW_OBJECT_HARDWARE_CONFIG,
    {
        'groups': {
            'guide': {
                'motor_ids': [60],
                'calib_scale': [1],
                'calib_offset': [-np.pi],
            }
        },
    },
)
