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

"""Utility functions relating to math."""

from typing import Iterable

import numpy as np


def average_quaternions(quaternions: Iterable[np.ndarray]) -> np.ndarray:
    """Returns the average of the given quaternions.

    Args:
        quaternions: A list of quaternions to average.

    Returns:
        The averaged quaternion.
    """
    # Implements the algorithm from:
    # Markley, F. L., Cheng, Y., Crassidis, J. L., & Oshman, Y. (2007).
    # Averaging quaternions. Journal of Guidance, Control, and Dynamics,
    # 30(4), 1193-1197.
    n_quat = len(quaternions)
    assert n_quat > 0, 'Must provide at least one quaternion.'
    weight = 1.0 / n_quat  # Uniform weighting for all quaternions.
    q_matrix = np.vstack(quaternions)
    assert q_matrix.shape == (n_quat, 4)
    m_matrix = np.matmul(weight * np.transpose(q_matrix), q_matrix)
    _, eig_vecs = np.linalg.eigh(m_matrix)
    # The final eigenvector corresponds to the largest eigenvalue.
    return eig_vecs[:, -1]
