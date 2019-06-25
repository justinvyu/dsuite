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

"""Simulated robot API backed by MuJoCo."""

import abc
import enum
from typing import Any, Union

from dsuite.simulation.renderer import Renderer


class SimBackend(enum.Enum):
    """Simulation library types."""
    MUJOCO_PY = 0
    DM_CONTROL = 1


class SimScene(metaclass=abc.ABCMeta):
    """Encapsulates a MuJoCo robotics simulation."""

    @staticmethod
    def create(*args, backend: Union[SimBackend, int], **kwargs) -> 'SimScene':
        """Creates a new simulation scene.

        Args:
            *args: Positional arguments to pass to the simulation.
            backend: The simulation backend to use to load the simulation.
            **kwargs: Keyword arguments to pass to the simulation.

        Returns:
            A SimScene object.
        """
        backend = SimBackend(backend)
        if backend == SimBackend.MUJOCO_PY:
            from dsuite.simulation.mjpy_sim_scene import MjPySimScene
            return MjPySimScene(*args, **kwargs)
        elif backend == SimBackend.DM_CONTROL:
            from dsuite.simulation.dm_sim_scene import DMSimScene
            return DMSimScene(*args, **kwargs)
        else:
            raise NotImplementedError(backend)

    def __init__(
            self,
            model_handle: Any,
            frame_skip: int = 1,
    ):
        """Initializes a new simulation.

        Args:
            model_handle: The simulation model to load. This can be a XML file,
                or a format/object specific to the simulation backend.
            frame_skip: The number of simulation steps per environment step.
                This multiplied by the timestep defined in the model file is the
                step duration.
        """
        self.frame_skip = frame_skip

        self.sim = self._load_simulation(model_handle)
        self.model = self.sim.model
        self.data = self.sim.data

        self.renderer = self._create_renderer(self.sim)

    @property
    def step_duration(self):
        """Returns the simulation step duration in seconds."""
        return self.model.opt.timestep * self.frame_skip

    def close(self):
        """Cleans up any resources used by the simulation."""
        self.renderer.close()

    def advance(self):
        """Advances the simulation for one step."""
        # Step the simulation `frame_skip` times.
        for _ in range(self.frame_skip):
            self.sim.step()
            self.renderer.refresh_window()

    @abc.abstractmethod
    def copy_model(self) -> Any:
        """Returns a copy of the MjModel object."""

    @abc.abstractmethod
    def save_binary(self, path: str) -> str:
        """Saves the loaded model to a binary .mjb file.

        Returns:
            The file path that the binary was saved to.
        """

    @abc.abstractmethod
    def _load_simulation(self, model_handle: Any) -> Any:
        """Loads the simulation from the given model handle."""

    @abc.abstractmethod
    def _create_renderer(self, sim: Any) -> Renderer:
        """Creates a renderer for the given simulation."""

    @abc.abstractmethod
    def _get_mjlib(self) -> Any:
        """Returns an interface to the low-level MuJoCo API."""
