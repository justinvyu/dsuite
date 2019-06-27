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
[DynamixelSDK](https://github.com/ROBOTIS-GIT/DynamixelSDK).

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

Download MuJoCo Pro 2.00 from the
[MuJoCo website](https://www.roboti.us/index.html). You should extract this
to `~/.mujoco/mujoco200`. Ensure your MuJoCo license key is placed at
`~/.mujoco/mjkey.txt`.

#### D'Suite

D'Suite requires Python 3.5 or higher. You can install D'Suite by running:

```bash
pip install dsuite
```

We recommend doing this in a `virtualenv` or with the `--user` flag to avoid
interfering with system packages.

## Disclaimer

This is not an official Google product.
