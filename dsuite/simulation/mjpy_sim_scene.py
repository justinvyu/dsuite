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

"""Simulation using DeepMind Control Suite."""

import os
from typing import Any

import mujoco_py

from dsuite.simulation.mjpy_renderer import MjPyRenderer
from dsuite.simulation.sim_scene import SimScene


class MjPySimScene(SimScene):
    """Encapsulates a MuJoCo robotics simulation using mujoco_py."""

    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle.

        Args:
            model_handle: Path to the Mujoco XML file to load.

        Returns:
            A mujoco_py MjSim object.
        """
        if isinstance(model_handle, str):
            if not os.path.isfile(model_handle):
                raise ValueError(
                    '[MjPySimScene] Invalid model file path: {}'.format(
                        model_handle))

            model = mujoco_py.load_model_from_path(model_handle)
            sim = mujoco_py.MjSim(model)
        else:
            raise NotImplementedError(model_handle)
        return sim

    def _create_renderer(self, sim: Any) -> MjPyRenderer:
        """Creates a renderer for the given simulation."""
        return MjPyRenderer(sim)

    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""
        null_model = self._get_mjlib().PyMjModel()
        model_copy = self._get_mjlib().mj_copyModel(null_model, self.model)
        return model_copy

    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """
        if not path.endswith('.mjb'):
            path = path + '.mjb'
        self._get_mjlib().mj_saveModel(self.model.ptr, path.encode(), None, 0)
        return path

    def _get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""
        return _MjlibWrapper(mujoco_py.cymj)


class _MjlibWrapper:
    """Wrapper that forwards mjlib calls."""

    def __init__(self, lib):
        self._lib = lib

    def __getattr__(self, name: str):
        if name.startswith('mj'):
            return getattr(self._lib, '_' + name)
        return getattr(self._lib, name)
