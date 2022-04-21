# D'Suite

D'Suite is a set of reinforcement learning environments for learning dexterous
robotics tasks. The environments can be run in simulation or on real hardware
with low-cost, general hardware.

The hardware design and build instructions are fully open-sourced and are
available for anyone to build their own robots.

## Features

D'Suite environments are fully [Gym](https://gym.openai.com/)-compliant and can
be used with any reinforcement learning library that interfaces with Gym
environments.

Simulation is performed by [MuJoCo](http://www.mujoco.org/). Communication with
hardware is done through the
[DynamixelSDK](https://github.com/ROBOTIS-GIT/DynamixelSDK). For D'Kitty
environments, external tracking is supported through
[OpenVR](https://github.com/ValveSoftware/openvr) tracking.

```python
import dsuite
import gym

# Create a simulation environment for the D'Claw turning task.
env = gym.make('DClawTurnFixed-v0')

# Reset the environent and perform a random action.
env.reset()
env.step(env.action_space.sample())

# Create a hardware environment for the D'Claw turning task.
# `device_path` refers to the device port of the Dynamixel USB device.
# e.g. '/dev/ttyUSB0' for Linux, '/dev/tty.usbserial-*' for Mac OS.
env = gym.make('DClawTurnFixed-v0', device_path='/dev/ttyUSB0')

# The hardware environment has the same interface as the simulation environment.
env.reset()
env.step(env.action_space.sample())
```

## Installation

#### MuJoCo

Download MuJoCo 2.1.0 from the
[MuJoCo repo](https://github.com/deepmind/mujoco/releases). You should extract this
to `~/.mujoco/mujoco210`. No license needed anymore!

#### D'Suite

D'Suite requires Python 3.5 or higher. You can install D'Suite by running the following command in this directory:

```bash
pip install -e .
```

Then, clone ``dsuite-scenes`` into this directory, and checkout the `dev` branch on the repo using:

```bash
git clone -b dev dsuite-scenes
```

#### Visualizing Environments

Find a full list of registered environments in `dsuite/dclaw/__init__.py`.

Visualize and save the video of one of these environments, running a random policy using this script:

```bash
python -m dsuite.scripts.examine_random_policy -d DClawTurnFreeValve3 -e Fixed-v0 -f ./video.mp4
```

## Disclaimer

This is not an official Google product.

