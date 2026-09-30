#!/usr/bin/env python3
"""Convert Gazebo downward ranges into Pinky Pro-style raw IR ADC data."""

import math
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Range
from std_msgs.msg import UInt16MultiArray


def range_sees_table(distance, sensor_minimum, sensor_maximum,
                     tabletop_maximum):
    """Return whether a simulated downward ray has a short valid return."""
    return (
        math.isfinite(distance)
        and sensor_minimum <= distance <= sensor_maximum
        and distance <= tabletop_maximum
    )


def simulated_adc_value(on_surface, surface_value, off_surface_value,
                        noise=0.0):
    """Return one clamped 12-bit ADC value for a simulated IR channel."""
    base_value = surface_value if on_surface else off_surface_value
    return min(4095, max(0, int(round(float(base_value) + float(noise)))))


class IrAdcSim(Node):
    """Publish the same IR topic and message type as Pinky Pro hardware."""

    SENSOR_ORDER = ('left', 'center', 'right')

    def __init__(self):
        super().__init__('ir_adc_sim')
        self._declare_parameters()

        self._adc_values = {sensor: None for sensor in self.SENSOR_ORDER}
        self._last_update = {sensor: None for sensor in self.SENSOR_ORDER}
        seed = int(self.get_parameter('random_seed').value)
        self._random = random.Random(seed)

        for sensor in self.SENSOR_ORDER:
            topic = str(self.get_parameter(f'{sensor}_range_topic').value)
            self.create_subscription(
                Range,
                topic,
                lambda msg, position=sensor: self._range_callback(
                    msg, position),
                qos_profile_sensor_data,
            )

        adc_topic = str(self.get_parameter('adc_topic').value)
        self._publisher = self.create_publisher(
            UInt16MultiArray, adc_topic, qos_profile_sensor_data)
        publish_rate = max(
            1.0, float(self.get_parameter('publish_rate').value))
        self._timer = self.create_timer(1.0 / publish_rate, self._publish)

        self.get_logger().info(
            'Gazebo IR ADC emulator ready: '
            f'{self.SENSOR_ORDER} -> {adc_topic}. '
            'Channel order is [left, center, right].')

    def _declare_parameters(self):
        """Declare simulator input, output, and ADC-model parameters."""
        self.declare_parameter('left_range_topic', '/ir/left')
        self.declare_parameter('center_range_topic', '/ir/center')
        self.declare_parameter('right_range_topic', '/ir/right')
        self.declare_parameter('adc_topic', '/ir_sensor/range')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('input_timeout', 0.3)
        self.declare_parameter('tabletop_max_distance', 0.05)
        self.declare_parameter('surface_adc_values', [3200, 3200, 3200])
        self.declare_parameter('off_surface_adc_values', [100, 100, 100])
        self.declare_parameter('adc_noise_stddev', 12.0)
        self.declare_parameter('random_seed', 7)

    def _range_callback(self, msg, sensor):
        """Convert one Gazebo Range sample and remember its receive time."""
        surface_values = self._integer_array('surface_adc_values')
        off_surface_values = self._integer_array('off_surface_adc_values')
        index = self.SENSOR_ORDER.index(sensor)
        sees_table = range_sees_table(
            float(msg.range),
            float(msg.min_range),
            float(msg.max_range),
            float(self.get_parameter('tabletop_max_distance').value),
        )
        noise = self._random.gauss(
            0.0, max(0.0, float(
                self.get_parameter('adc_noise_stddev').value)))
        self._adc_values[sensor] = simulated_adc_value(
            sees_table,
            surface_values[index],
            off_surface_values[index],
            noise,
        )
        self._last_update[sensor] = self.get_clock().now()

    def _publish(self):
        """Publish only when all three simulated sensors are fresh."""
        now = self.get_clock().now()
        timeout = max(
            0.01, float(self.get_parameter('input_timeout').value))
        if any(
            value is None
            or self._last_update[sensor] is None
            or (now - self._last_update[sensor]).nanoseconds / 1e9 > timeout
            for sensor, value in self._adc_values.items()
        ):
            return

        message = UInt16MultiArray()
        message.data = [
            self._adc_values[sensor] for sensor in self.SENSOR_ORDER]
        self._publisher.publish(message)

    def _integer_array(self, name):
        """Read and validate one three-channel integer parameter."""
        values = [int(value) for value in self.get_parameter(name).value]
        if len(values) != 3:
            raise ValueError(f'{name} must contain exactly three values')
        return values


def main(args=None):
    """Run the Gazebo-to-ADC conversion node."""
    rclpy.init(args=args)
    node = IrAdcSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
