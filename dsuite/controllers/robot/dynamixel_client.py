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

"""Communication using the DynamixelSDK."""

import logging
from typing import Iterable, Optional, Union, Tuple

import numpy as np

PROTOCOL_VERSION = 2.0

# The following addresses assume XH motors.
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POS_VEL = 128

# Data Byte Length
LEN_PRESENT_POSITION = 4
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_POS_VEL = 8
LEN_GOAL_POSITION = 4

POS_SCALE = 2. * np.pi / 4096  # 0.088 degrees
VEL_SCALE = 0.11 * 2. * np.pi / 60  # 0.11 rpm


class DynamixelClient:
    """Client for communicating with Dynamixel motors.

    NOTE: This only supports Protocol 2.
    """

    def __init__(self,
                 motor_ids: Iterable[int],
                 port: str = '/dev/ttyUSB0',
                 baudrate: int = 1000000,
                 lazy_connect: bool = False):
        """Initializes a new client.

        Args:
            motor_ids: All motor IDs being used by the client.
            port: The Dynamixel device to talk to. e.g.
                - Linux: /dev/ttyUSB0
                - Mac: /dev/tty.usbserial-*
                - Windows: COM1
            baudrate: The Dynamixel baudrate to communicate with.
            lazy_connect: If True, automatically connects when calling a method
                that requires a connection, if not already connected.
        """
        import dynamixel_sdk
        self.dxl = dynamixel_sdk

        self.motor_ids = list(motor_ids)
        self.port_name = port
        self.baudrate = baudrate
        self.lazy_connect = lazy_connect

        self.port_handler = self.dxl.PortHandler(port)
        self.packet_handler = self.dxl.PacketHandler(PROTOCOL_VERSION)

        self._pos_vel_reader = DynamixelPosVelReader(self, self.motor_ids)
        self._sync_writers = {}
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self):
        """Connects to the Dynamixel motors.

        NOTE: This should be called after all DynamixelClients on the same
            process are created.
        """
        assert not self._is_connected, 'Client is already connected.'

        if self.port_handler.openPort():
            logging.info('Succeeded to open port: %s', self.port_name)
        else:
            raise OSError(
                ('Failed to open port at {} (Check that the device is powered '
                 'on and connected to your computer).').format(self.port_name))

        if self.port_handler.setBaudRate(self.baudrate):
            logging.info('Succeeded to set baudrate to %d', self.baudrate)
        else:
            raise OSError(
                ('Failed to set the baudrate to {} (Ensure that the device was '
                 'configured for this baudrate).').format(self.baudrate))
        self._is_connected = True

        # Start with all motors disabled.
        self.set_torque_enabled(self.motor_ids, False)

    def disconnect(self):
        if not self._is_connected:
            return
        # Ensure motors are disabled at the end.
        self.set_torque_enabled(self.motor_ids, False)
        self.port_handler.closePort()
        self._is_connected = False

    def set_torque_enabled(self, motor_ids: Iterable[int], enabled: bool):
        """Sets whether torque is enabled for the motors.

        Args:
            motor_ids: The motor IDs to configure.
            enabled: Whether to engage or disengage the motors.
        """
        errored_ids = self.write_byte(
            motor_ids,
            int(enabled),
            ADDR_TORQUE_ENABLE,
        )
        if errored_ids:
            logging.error('Could not set torque enabled for IDs: %s',
                          str(errored_ids))

    def read_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the current positions and velocities."""
        return self._pos_vel_reader.read()

    def write_desired_pos(self, motor_ids: Iterable[int],
                          positions: np.ndarray):
        """Writes the given desired positions.

        Args:
            motor_ids: The motor IDs to write to.
            positions: The joint angles in radians to write.
        """
        assert len(motor_ids) == len(positions)

        # Convert to Dynamixel position space.
        positions = positions / POS_SCALE
        self.sync_write(motor_ids, positions, ADDR_GOAL_POSITION,
                        LEN_GOAL_POSITION)

    def write_byte(
            self,
            motor_ids: Iterable[int],
            value: int,
            address: int,
    ) -> Iterable[int]:
        """Writes a value to the motors.

        Args:
            motor_ids: The motor IDs to write to.
            value: The value to write to the control table.
            address: The control table address to write to.

        Returns:
            A list of IDs that were unsuccessful.
        """
        self.check_connected()
        errored_ids = []
        for motor_id in motor_ids:
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, address, value)
            success = self.handle_packet_result(comm_result, dxl_error,
                                                motor_id)
            if not success:
                errored_ids.append(motor_id)
        return errored_ids

    def sync_write(self, motor_ids: Iterable[int],
                   values: Iterable[Union[int, float]], address: int,
                   size: int):
        """Writes values to a group of motors.

        Args:
            motor_ids: The motor IDs to write to.
            values: The values to write.
            address: The control table address to write to.
            size: The size of the control table value being written to.
        """
        self.check_connected()
        key = (address, size)
        if key not in self._sync_writers:
            self._sync_writers[key] = self.dxl.GroupSyncWrite(
                self.port_handler, self.packet_handler, address, size)
        sync_writer = self._sync_writers[key]

        errored_ids = []
        for motor_id, desired_pos in zip(motor_ids, values):
            # Pack the goal value into a little-endian byte array.
            value_bytes = int(desired_pos).to_bytes(size, byteorder='little')
            success = sync_writer.addParam(motor_id, value_bytes)
            if not success:
                errored_ids.append(motor_id)

        if errored_ids:
            logging.error('Sync write failed for: %s', str(errored_ids))

        comm_result = sync_writer.txPacket()
        self.handle_packet_result(comm_result)

        sync_writer.clearParam()

    def check_connected(self):
        """Ensures the robot is connected."""
        if self.lazy_connect and not self._is_connected:
            self.connect()
        if not self._is_connected:
            raise OSError('Must call connect() first.')

    def handle_packet_result(self,
                             comm_result: int,
                             dxl_error: Optional[int] = None,
                             dxl_id: Optional[int] = None):
        """Handles the result from a communication request."""
        error_message = None
        if comm_result != self.dxl.COMM_SUCCESS:
            error_message = self.packet_handler.getTxRxResult(comm_result)
        elif dxl_error is not None:
            error_message = self.packet_handler.getRxPacketError(dxl_error)
        if error_message:
            if dxl_id is not None:
                error_message = (
                    '[Motor ID: {}] '.format(dxl_id) + error_message)
            logging.error(error_message)
            return False
        return True

    def __enter__(self):
        """Enables use as a context manager."""
        if not self._is_connected:
            self.connect()
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.disconnect()

    def __del__(self):
        """Automatically disconnect on destruction."""
        if self._is_connected:
            self.disconnect()


class DynamixelReader:
    """Reads data from Dynamixel motors.

    This wraps a GroupBulkRead from the DynamixelSDK.
    """

    def __init__(self, client: DynamixelClient, motor_ids: Iterable[int],
                 address: int, size: int):
        """Initializes a new reader."""
        self.client = client
        self.motor_ids = motor_ids
        self.address = address
        self.size = size
        self._initialize_data()

        self.operation = self.client.dxl.GroupBulkRead(client.port_handler,
                                                       client.packet_handler)

        for motor_id in motor_ids:
            success = self.operation.addParam(motor_id, address, size)
            if not success:
                raise OSError(
                    '[Motor ID: {}] Could not add parameter to bulk read.'
                    .format(motor_id))

    def read(self, retries: int = 1):
        """Reads data from the motors."""
        self.client.check_connected()
        success = False
        while not success and retries >= 0:
            comm_result = self.operation.txRxPacket()
            success = self.client.handle_packet_result(comm_result)
            retries -= 1

        # If we failed, send a copy of the previous data.
        if not success:
            return self._get_data()

        errored_ids = []
        for i, motor_id in enumerate(self.motor_ids):
            # Check if the data is available.
            available = self.operation.isAvailable(motor_id, self.address,
                                                   self.size)
            if not available:
                errored_ids.append(motor_id)
                continue

            self._update_data(i, motor_id)

        if errored_ids:
            logging.error('Bulk read data is unavailable for: %s',
                          str(errored_ids))

        return self._get_data()

    def _initialize_data(self):
        """Initializes the cached data."""
        self._data = np.zeros(len(self.motor_ids), dtype=np.float32)

    def _update_data(self, index: int, motor_id: int):
        """Updates the data index for the given motor ID."""
        self._data[index] = self.operation.getData(motor_id, self.address,
                                                   self.size)

    def _get_data(self):
        """Returns a copy of the data."""
        return self._data.copy()


class DynamixelPosVelReader(DynamixelReader):
    """Reads positions and velocities."""

    def __init__(self, client: DynamixelClient, motor_ids: Iterable[int]):
        super().__init__(
            client,
            motor_ids,
            address=ADDR_PRESENT_POS_VEL,
            size=LEN_PRESENT_POS_VEL,
        )

    def _initialize_data(self):
        """Initializes the cached data."""
        self._pos_data = np.zeros(len(self.motor_ids), dtype=np.float32)
        self._vel_data = np.zeros(len(self.motor_ids), dtype=np.float32)

    def _update_data(self, index: int, motor_id: int):
        """Updates the data index for the given motor ID."""
        vel = self.operation.getData(motor_id, ADDR_PRESENT_VELOCITY,
                                     LEN_PRESENT_VELOCITY)
        pos = self.operation.getData(motor_id, ADDR_PRESENT_POSITION,
                                     LEN_PRESENT_POSITION)
        if vel >= 1024:
            vel = -(vel - 1024)

        self._pos_data[index] = pos * POS_SCALE
        self._vel_data[index] = vel * VEL_SCALE

    def _get_data(self):
        """Returns a copy of the data."""
        return self._pos_data.copy(), self._vel_data.copy()
