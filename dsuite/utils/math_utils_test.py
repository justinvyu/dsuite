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

"""Tests for math_utils."""

import unittest

import numpy as np
from transforms3d.euler import euler2quat, quat2euler

from dsuite.utils.math_utils import average_quaternions


class AverageQuaternionsTest(unittest.TestCase):
    """Tests for `average_quaternions`."""

    def test_identity(self):
        """Average one quaternion should equal itself."""
        test_quat = euler2quat(np.pi / 4, np.pi / 4, np.pi / 4)
        avg_quat = average_quaternions([test_quat])
        np.testing.assert_array_almost_equal(avg_quat, test_quat)

    def test_multiple_identity(self):
        """Average multiple copies of a quaternion should equal itself."""
        test_quat = euler2quat(np.pi / 4, np.pi / 4, np.pi / 4)
        avg_quat = average_quaternions([test_quat, test_quat, test_quat])
        np.testing.assert_array_almost_equal(avg_quat, test_quat)

    def test_average_two(self):
        """Averaging two different quaternions."""
        quat1 = euler2quat(np.pi / 4, 0, 0)
        quat2 = euler2quat(-np.pi / 4, 0, 0)
        avg_quat = average_quaternions([quat1, quat2])
        result = quat2euler(avg_quat)
        np.testing.assert_array_almost_equal(result, [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
