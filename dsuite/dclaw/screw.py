# Copyright 2019 The DSuite Authors.
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

"""Screw tasks with DClaw robots.

This is continuous rotation of an object to match a target velocity.
"""

import numpy as np

from dsuite.dclaw.turn import BaseDClawTurn
from dsuite.utils.configurable import configurable


class BaseDClawScrew(BaseDClawTurn):
    """Shared logic for DClaw screw tasks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # The target velocity is set during `_reset`.
        self._target_object_vel = 0

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        super()._step(action)

        # Update the target object goal.
        self._set_target_object_pos(self._target_object_pos +
                                    self._target_object_vel * self.dt)


@configurable(pickleable=True)
class DClawScrewFixed(BaseDClawScrew):
    """Rotates the object with a fixed initial position and velocity."""

    def _reset(self):
        # Start from the target and rotate at a constant velocity.
        self._initial_object_pos = 0
        self._set_target_object_pos(0)
        self._target_object_vel = 0.5
        super()._reset()


@configurable(pickleable=True)
class DClawScrewRandom(BaseDClawScrew):
    """Rotates the object with a random initial position and velocity."""

    def _reset(self):
        # Initial position is +/- 180 degrees.
        self._initial_object_pos = self.np_random.uniform(
            low=-np.pi, high=np.pi)
        self._set_target_object_pos(self._initial_object_pos)

        # Random target velocity.
        self._target_object_vel = self.np_random.uniform(low=-0.75, high=0.75)
        super()._reset()


@configurable(pickleable=True)
class DClawScrewRandomDynamics(DClawScrewRandom):
    """Rotates the object with a random initial position and velocity.

    The dynamics of the simulation are randomized each episode.
    """

    def _reset(self):
        self._randomize_claw_sim()
        self._randomize_object_sim()
        super()._reset()
