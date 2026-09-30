"""Unit tests for math used by the Pinky tabletop patrol."""

import math
from types import SimpleNamespace

from pinky_move.ir_adc_sim_node import (
    range_sees_table,
    simulated_adc_value,
)
from pinky_move.move_node import (
    absolute_leg_heading,
    ir_adc_channels_on_surface,
    normalize_angle,
    path_cross_track_error,
    path_heading_with_correction,
    projected_progress,
    usable_laser_range,
    yaw_from_quaternion,
)
import pytest


@pytest.mark.parametrize(
    'source, expected',
    [
        (0.0, 0.0),
        (2.0 * math.pi, 0.0),
        (3.0 * math.pi, -math.pi),
        (-3.0 * math.pi, -math.pi),
    ],
)
def test_normalize_angle(source, expected):
    """Angles wrap into the controller's canonical interval."""
    assert normalize_angle(source) == pytest.approx(expected)


def test_positive_infinity_means_no_lidar_return():
    """Positive infinity is clear up to the sensor maximum range."""
    assert usable_laser_range(math.inf, 0.05, 12.0) == 12.0


def test_invalid_lidar_ranges_are_rejected():
    """NaN, negative infinity, and out-of-range samples are unsafe."""
    assert usable_laser_range(math.nan, 0.05, 12.0) is None
    assert usable_laser_range(-math.inf, 0.05, 12.0) is None
    assert usable_laser_range(0.01, 0.05, 12.0) is None


def test_all_three_ir_adc_channels_must_match_calibration():
    """Every ADC channel must remain inside its own tabletop window."""
    assert ir_adc_channels_on_surface(
        [1200, 1800, 2400],
        [1000, 1600, 2200],
        [1400, 2000, 2600],
    )
    assert not ir_adc_channels_on_surface(
        [1200, 1800, 3000],
        [1000, 1600, 2200],
        [1400, 2000, 2600],
    )


def test_malformed_or_unusable_ir_adc_calibration_is_unsafe():
    """Missing channels and inverted bounds fail closed."""
    assert not ir_adc_channels_on_surface(
        [1200, 1800], [1000, 1600], [1400, 2000])
    assert not ir_adc_channels_on_surface(
        [1200, 1800, 2400],
        [1400, 1600, 2200],
        [1000, 2000, 2600],
    )


def test_downward_ir_range_requires_a_short_finite_return():
    """The ADC emulator accepts only a short finite Gazebo return."""
    assert range_sees_table(0.013, 0.002, 0.30, 0.05)
    assert not range_sees_table(0.08, 0.002, 0.30, 0.05)
    assert not range_sees_table(math.inf, 0.002, 0.30, 0.05)


def test_simulated_adc_uses_surface_state_and_clamps_to_twelve_bits():
    """Gazebo table/no-table states become Pinky-style 12-bit ADC values."""
    assert simulated_adc_value(True, 3200, 100, -10) == 3190
    assert simulated_adc_value(False, 3200, 100, 5) == 105
    assert simulated_adc_value(True, 4090, 100, 20) == 4095
    assert simulated_adc_value(False, 3200, 3, -20) == 0


def test_yaw_from_quaternion_for_ninety_degrees():
    """Quaternion conversion returns the expected planar yaw."""
    half_angle = math.pi / 4.0
    quaternion = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(half_angle),
        w=math.cos(half_angle),
    )
    assert yaw_from_quaternion(quaternion) == pytest.approx(math.pi / 2.0)


def test_absolute_leg_headings_do_not_accumulate_turn_error():
    """Every return uses the original heading, not the last imperfect yaw."""
    reference = math.radians(3.0)
    assert absolute_leg_heading(
        reference, 1, math.pi) == pytest.approx(reference)
    assert absolute_leg_heading(
        reference, -1, math.pi) == pytest.approx(
            normalize_angle(reference + math.pi))


def test_projected_progress_does_not_count_sideways_drift():
    """Only motion along the intended leg advances its distance counter."""
    assert projected_progress((0.0, 0.0), (1.0, 0.2), 0.0) == pytest.approx(
        1.0)


def test_cross_track_correction_points_back_toward_route():
    """Both travel directions steer a positive lateral error toward zero."""
    error = path_cross_track_error((0.0, 0.0), (0.5, 0.1), 0.0)
    assert error == pytest.approx(0.1)
    outbound = path_heading_with_correction(
        0.0, 1, error, 2.0, math.radians(10.0))
    returning = path_heading_with_correction(
        -math.pi, -1, error, 2.0, math.radians(10.0))
    assert outbound == pytest.approx(math.radians(-10.0))
    assert returning == pytest.approx(math.radians(-170.0))
