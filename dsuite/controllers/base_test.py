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

"""Unit tests for BaseController."""

import unittest
from parameterized import parameterized

from dsuite.controllers.base import BaseController
from dsuite.utils.testing.mock_sim_scene import MockSimScene


class DummyController(BaseController):
    """Mock controller for testing BaseController."""

    def __init__(self, **kwargs):
        super().__init__(sim_scene=MockSimScene(nq=1), **kwargs)

    def _process_group(self, **config_kwargs):
        return {}

    def _get_group_states(self, configs):
        return [0 for group in configs]


class BaseControllerTest(unittest.TestCase):
    """Unit test class for BaseController."""

    def test_get_state(self):
        """Tests retrieving state from a single group."""
        controller = DummyController(groups={'foo': {}})
        state = controller.get_state('foo')
        self.assertEqual(state, 0)

    def test_get_states(self):
        """Tests retrieving state from multiple groups."""
        controller = DummyController(groups={'foo': {}, 'bar': {}})
        foo_state, bar_state = controller.get_state(['foo', 'bar'])
        self.assertEqual(foo_state, 0)
        self.assertEqual(bar_state, 0)

    @parameterized.expand([
        (3, 1, None, [0]),
        (3, 1, 2, [0]),
        (3, 3, None, [[0], [0], [0]]),
    ])
    def test_get_past_states(self, state_history_len, n, i, output):
        """Add states to the history, checking length."""
        controller = DummyController(
            groups={'foo': {}}, state_history_len=state_history_len)
        # State history should be initially empty
        self.assertIsNotNone(controller._state_history)
        self.assertEqual(len(controller._state_history), 0)
        # Add items to the state history by calling get_state()
        for _ in range(state_history_len + 2):
            controller.get_state(['foo'])
        self.assertEqual(len(controller._state_history), state_history_len)
        # Get items from state history
        history = controller.get_past_states(n, i)
        self.assertEqual(history, output)


if __name__ == '__main__':
    unittest.main()
