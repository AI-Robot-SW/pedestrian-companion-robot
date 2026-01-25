"""
Tests for LocationSensor.
"""

import asyncio
import time
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.location_sensor import (
    LocationData,
    LocationSensor,
    LocationSensorConfig,
    NavigationGoal,
    Pose2D,
    Pose3D,
)


@pytest.fixture
def config():
    """Configuration for LocationSensor."""
    return LocationSensorConfig(
        localization_method="lidar",
        update_rate_hz=10.0,
        min_confidence_threshold=0.5,
        enable_location_names=True,
        location_map_file=None,
        coordinate_frame="map",
        report_interval_ms=1000,
    )


@pytest.fixture
def config_custom():
    """Custom configuration for LocationSensor."""
    return LocationSensorConfig(
        localization_method="amcl",
        update_rate_hz=20.0,
        min_confidence_threshold=0.7,
        enable_location_names=False,
        coordinate_frame="odom",
        report_interval_ms=500,
    )


class TestPose2D:
    """Test Pose2D dataclass."""

    def test_pose2d_creation(self):
        """Test Pose2D creation."""
        pose = Pose2D(x=1.0, y=2.0, theta=0.5)

        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.theta == 0.5


class TestPose3D:
    """Test Pose3D dataclass."""

    def test_pose3d_creation(self):
        """Test Pose3D creation."""
        pose = Pose3D(x=1.0, y=2.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)

        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 0.0
        assert pose.qw == 1.0


class TestNavigationGoal:
    """Test NavigationGoal dataclass."""

    def test_navigation_goal_creation(self):
        """Test NavigationGoal creation."""
        pose = Pose2D(x=5.0, y=5.0, theta=0.0)
        goal = NavigationGoal(
            goal_id="goal_1",
            name="회의실",
            pose=pose,
            is_active=True,
        )

        assert goal.goal_id == "goal_1"
        assert goal.name == "회의실"
        assert goal.pose.x == 5.0
        assert goal.is_active is True

    def test_navigation_goal_defaults(self):
        """Test NavigationGoal default values."""
        pose = Pose2D(x=0.0, y=0.0, theta=0.0)
        goal = NavigationGoal(goal_id="goal_1", name=None, pose=pose)

        assert goal.name is None
        assert goal.is_active is False


class TestLocationData:
    """Test LocationData dataclass."""

    def test_location_data_creation(self):
        """Test LocationData creation."""
        pose = Pose2D(x=1.0, y=2.0, theta=0.5)
        data = LocationData(
            timestamp=time.time(),
            current_pose=pose,
            current_location_name="로비",
            is_navigating=False,
            navigation_status="idle",
        )

        assert data.current_pose.x == 1.0
        assert data.current_location_name == "로비"
        assert data.is_navigating is False

    def test_location_data_defaults(self):
        """Test LocationData default values."""
        data = LocationData(timestamp=time.time())

        assert data.current_pose is None
        assert data.current_goal is None
        assert data.distance_to_goal is None
        assert data.localization_confidence == 0.0
        assert data.is_navigating is False
        assert data.navigation_status == "idle"


class TestLocationSensorConfig:
    """Test LocationSensorConfig initialization."""

    def test_config_default_values(self):
        """Test LocationSensorConfig default values."""
        config = LocationSensorConfig()

        assert config.localization_method == "lidar"
        assert config.update_rate_hz == 10.0
        assert config.min_confidence_threshold == 0.5
        assert config.enable_location_names is True
        assert config.coordinate_frame == "map"
        assert config.report_interval_ms == 1000

    def test_config_custom_values(self):
        """Test LocationSensorConfig with custom values."""
        config = LocationSensorConfig(
            localization_method="amcl",
            update_rate_hz=20.0,
            min_confidence_threshold=0.8,
        )

        assert config.localization_method == "amcl"
        assert config.update_rate_hz == 20.0
        assert config.min_confidence_threshold == 0.8


class TestLocationSensorInitialization:
    """Test LocationSensor initialization."""

    def test_sensor_initialization(self, config):
        """Test LocationSensor initialization."""
        sensor = LocationSensor(config=config)

        assert sensor.descriptor_for_LLM == "Location"
        assert isinstance(sensor.message_buffer, Queue)
        assert sensor.latest_location is None

    def test_sensor_config_access(self, config):
        """Test sensor has access to config."""
        sensor = LocationSensor(config=config)

        assert sensor.config == config
        assert sensor.config.localization_method == "lidar"


class TestLocationSensorCallback:
    """Test LocationSensor location update callback."""

    def test_handle_location_update(self, config):
        """Test handling location update."""
        sensor = LocationSensor(config=config)
        pose = Pose2D(x=1.0, y=2.0, theta=0.0)
        location_data = LocationData(
            timestamp=time.time(),
            current_pose=pose,
            navigation_status="idle",
        )

        sensor._handle_location_update(location_data)

        assert sensor.latest_location is location_data
        assert sensor.message_buffer.qsize() == 1


class TestLocationSensorPoll:
    """Test LocationSensor polling."""

    @pytest.mark.asyncio
    async def test_poll_with_data(self, config):
        """Test polling when data is available."""
        sensor = LocationSensor(config=config)
        location_data = LocationData(
            timestamp=time.time(),
            navigation_status="idle",
        )
        sensor.message_buffer.put(location_data)

        result = await sensor._poll()

        assert result is not None
        assert result.navigation_status == "idle"

    @pytest.mark.asyncio
    async def test_poll_empty_buffer(self, config):
        """Test polling when buffer is empty."""
        sensor = LocationSensor(config=config)

        result = await sensor._poll()

        assert result is None


class TestLocationSensorRawToText:
    """Test LocationSensor raw to text conversion."""

    @pytest.mark.asyncio
    async def test_raw_to_text_with_location_name(self, config):
        """Test converting LocationData with location name."""
        sensor = LocationSensor(config=config)
        location_data = LocationData(
            timestamp=time.time(),
            current_location_name="회의실",
        )

        message = await sensor._raw_to_text(location_data)

        assert message is not None
        assert "회의실" in message.message

    @pytest.mark.asyncio
    async def test_raw_to_text_with_pose(self, config):
        """Test converting LocationData with pose."""
        sensor = LocationSensor(config=config)
        pose = Pose2D(x=1.5, y=2.5, theta=0.0)
        location_data = LocationData(
            timestamp=time.time(),
            current_pose=pose,
        )

        message = await sensor._raw_to_text(location_data)

        assert message is not None
        assert "1.50" in message.message
        assert "2.50" in message.message

    @pytest.mark.asyncio
    async def test_raw_to_text_navigating(self, config):
        """Test converting LocationData while navigating."""
        sensor = LocationSensor(config=config)
        goal_pose = Pose2D(x=5.0, y=5.0, theta=0.0)
        goal = NavigationGoal(goal_id="1", name="목적지", pose=goal_pose, is_active=True)
        location_data = LocationData(
            timestamp=time.time(),
            is_navigating=True,
            current_goal=goal,
            distance_to_goal=3.5,
        )

        message = await sensor._raw_to_text(location_data)

        assert message is not None
        assert "목적지" in message.message
        assert "3.5" in message.message

    @pytest.mark.asyncio
    async def test_raw_to_text_arrived(self, config):
        """Test converting LocationData when arrived."""
        sensor = LocationSensor(config=config)
        location_data = LocationData(
            timestamp=time.time(),
            navigation_status="arrived",
        )

        message = await sensor._raw_to_text(location_data)

        assert message is not None
        assert "Arrived" in message.message

    @pytest.mark.asyncio
    async def test_raw_to_text_none(self, config):
        """Test converting None input."""
        sensor = LocationSensor(config=config)

        message = await sensor._raw_to_text(None)

        assert message is None


class TestLocationSensorFormattedBuffer:
    """Test LocationSensor formatted buffer output."""

    def test_formatted_latest_buffer_empty(self, config):
        """Test formatted buffer when empty."""
        sensor = LocationSensor(config=config)

        result = sensor.formatted_latest_buffer()

        assert result is None

    def test_formatted_latest_buffer_with_location(self, config):
        """Test formatted buffer with location data."""
        sensor = LocationSensor(config=config)
        pose = Pose2D(x=1.0, y=2.0, theta=0.5)
        sensor.latest_location = LocationData(
            timestamp=time.time(),
            current_pose=pose,
            current_location_name="로비",
            navigation_status="idle",
        )

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Location" in result
        assert "로비" in result


class TestLocationSensorInterfaceMethods:
    """Test LocationSensor interface methods."""

    def test_get_current_pose(self, config):
        """Test getting current pose."""
        sensor = LocationSensor(config=config)
        pose = Pose2D(x=1.0, y=2.0, theta=0.5)
        sensor.latest_location = LocationData(
            timestamp=time.time(),
            current_pose=pose,
        )

        result = sensor.get_current_pose()

        assert result is not None
        assert result.x == 1.0
        assert result.y == 2.0

    def test_get_current_pose_none(self, config):
        """Test getting current pose when none available."""
        sensor = LocationSensor(config=config)

        result = sensor.get_current_pose()

        assert result is None

    def test_get_distance_to_goal(self, config):
        """Test getting distance to goal."""
        sensor = LocationSensor(config=config)
        sensor.latest_location = LocationData(
            timestamp=time.time(),
            distance_to_goal=5.5,
        )

        result = sensor.get_distance_to_goal()

        assert result == 5.5

    def test_has_active_goal_true(self, config):
        """Test has_active_goal returns True when goal is active."""
        sensor = LocationSensor(config=config)
        goal_pose = Pose2D(x=5.0, y=5.0, theta=0.0)
        goal = NavigationGoal(goal_id="1", name="목적지", pose=goal_pose, is_active=True)
        sensor.latest_location = LocationData(
            timestamp=time.time(),
            current_goal=goal,
        )

        result = sensor.has_active_goal()

        assert result is True

    def test_has_active_goal_false(self, config):
        """Test has_active_goal returns False when no active goal."""
        sensor = LocationSensor(config=config)

        result = sensor.has_active_goal()

        assert result is False

    def test_is_navigating(self, config):
        """Test is_navigating method."""
        sensor = LocationSensor(config=config)
        sensor.latest_location = LocationData(
            timestamp=time.time(),
            is_navigating=True,
        )

        result = sensor.is_navigating()

        assert result is True

    def test_get_navigation_status(self, config):
        """Test get_navigation_status method."""
        sensor = LocationSensor(config=config)
        sensor.latest_location = LocationData(
            timestamp=time.time(),
            navigation_status="navigating",
        )

        result = sensor.get_navigation_status()

        assert result == "navigating"

    def test_get_navigation_status_unknown(self, config):
        """Test get_navigation_status when no location data."""
        sensor = LocationSensor(config=config)

        result = sensor.get_navigation_status()

        assert result == "unknown"

    def test_get_localization_confidence(self, config):
        """Test get_localization_confidence method."""
        sensor = LocationSensor(config=config)
        sensor.latest_location = LocationData(
            timestamp=time.time(),
            localization_confidence=0.95,
        )

        result = sensor.get_localization_confidence()

        assert result == 0.95


class TestLocationSensorLifecycle:
    """Test LocationSensor start/stop lifecycle."""

    def test_start(self, config, caplog):
        """Test starting the sensor."""
        sensor = LocationSensor(config=config)

        sensor.start()

        assert "started" in caplog.text

    def test_stop(self, config, caplog):
        """Test stopping the sensor."""
        sensor = LocationSensor(config=config)

        sensor.stop()

        assert "stopped" in caplog.text
