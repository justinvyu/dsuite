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

"""Base API for Controllers.

A Controller provides a unified API between simulation and hardware.
"""

import abc
from collections import deque
import logging
from typing import Any, Dict, NewType, Sequence, Union

from dsuite.simulation.sim_scene import SimScene

# Type definition for a group configuration.
GroupConfig = NewType('GroupConfig', Any)
GroupState = NewType('GroupState', Any)


class BaseController(abc.ABC):
    """Base class for all controllers."""

    def __init__(
            self,
            sim_scene: SimScene,
            groups: Dict[str, Dict],
            state_history_len: int = 0,
    ):
        """Initializes a new controller.

        Args:
            sim_scene: The simulation to control.
            groups: Group configurations for reading/writing state.
            state_history_len: Maximum length for history of previous states.
        """
        self.sim_scene = sim_scene

        # Process all groups.
        self.groups = {}
        for group_name, group_config in groups.items():
            self.groups[group_name] = self._process_group(**group_config)

        # Keep track of state history along with relative timesteps
        self._state_history = None
        if state_history_len > 0:
            logging.debug('History of max length %s will be kept for %s',
                          state_history_len, self.__class__.__name__)
            self._state_history = deque(maxlen=state_history_len)

    def close(self):
        """Cleans up any resources used by the controller."""

    def get_state(
            self,
            groups: Union[str, Sequence[str]],
    ) -> Union[GroupState, Sequence[GroupState]]:
        """(Public) Returns the state of the given groups.

        Args:
            groups: Either a single group name or a list of group names of the
                groups to retrieve the state of.

        Returns:
            If `groups` is a string, returns a single state object. Otherwise,
            returns a list of state objects.
        """
        if isinstance(groups, str):
            states = self._get_group_states([self.get_config(groups)])
        else:
            states = self._get_group_states(
                [self.get_config(name) for name in groups])

        if self._state_history is not None:
            self._state_history.append(states)

        if isinstance(groups, str):
            return states[0]
        return states

    def get_past_states(
            self,
            n: int = 1,
            i: int = None,
    ) -> Union[GroupState, Sequence[GroupState]]:
        """Returns previous state or list of states from the state history.

        *Note* States are returned in reverse time order [newest, ... , oldest].
        Args:
            n: Number of previous states to return. If None, return the
                entire state history.
            i: If not None, ignore n and return the last i-th state.

        Returns:
            State object or list of state objects.
        """
        # Return a specific index of the state history
        if i is not None:
            # Clip to boundaries of state history
            clipped_i = max(0, min(i, len(self._state_history)))
            return self._state_history[-clipped_i]

        # Return the last single item in the state history
        if n == 1:
            return self._state_history[-1]

        # Return the last N items of state history (or all of it)
        history_list = list(self._state_history)
        history_list.reverse()
        if n is None:
            return history_list
        return history_list[:n]

    def get_config(self, group_name: str) -> GroupConfig:
        """Returns the configuration for a group."""
        if group_name not in self.groups:
            raise ValueError(
                'Group "{}" is not in the configured groups: {}'.format(
                    group_name, list(self.groups.keys())))
        return self.groups[group_name]

    @abc.abstractmethod
    def _process_group(self, **config_kwargs) -> GroupConfig:
        """Processes the configuration for a group.

        This should be overridden by subclasses to define and validate the group
        configuration.

        Args:
            **config_kwargs: Keyword arguments from the group configuration.

        Returns:
            An object that defines the group.
            e.g. A class that stores the group parameters.
        """

    @abc.abstractmethod
    def _get_group_states(
            self, configs: Sequence[GroupConfig]) -> Sequence[GroupState]:
        """(Private) Returns the states for the given group configurations."""
