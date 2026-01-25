"""
Tests for VisionSensor.
"""

import asyncio
import time
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.vision_sensor import (
    VisionData,
    VisionSensor,
    VisionSensorConfig,
)


@pytest.fixture
def config():
    """Configuration for VisionSensor."""
    return VisionSensorConfig(
        camera_device_id=None,
        camera_serial=None,
        width=640,
        height=480,
        fps=30,
        enable_depth=True,
        enable_segmentation=False,
        enable_bev=False,
        enable_pointcloud=False,
        vlm_model="gemini-pro-vision",
        vlm_prompt=None,
        inference_interval_ms=1000,
    )


@pytest.fixture
def config_custom():
    """Custom configuration for VisionSensor."""
    return VisionSensorConfig(
        camera_device_id=1,
        camera_serial="123456",
        width=1280,
        height=720,
        fps=60,
        enable_depth=True,
        enable_segmentation=True,
        enable_bev=True,
        enable_pointcloud=True,
        vlm_model="gpt-4-vision",
        vlm_prompt="Describe the scene",
        inference_interval_ms=500,
    )


class TestVisionSensorConfig:
    """Test VisionSensorConfig initialization."""

    def test_config_default_values(self):
        """Test VisionSensorConfig default values."""
        config = VisionSensorConfig()

        assert config.camera_device_id is None
        assert config.camera_serial is None
        assert config.width == 640
        assert config.height == 480
        assert config.fps == 30
        assert config.enable_depth is True
        assert config.enable_segmentation is False
        assert config.enable_bev is False
        assert config.enable_pointcloud is False
        assert config.vlm_model == "gemini-pro-vision"
        assert config.inference_interval_ms == 1000

    def test_config_custom_values(self):
        """Test VisionSensorConfig with custom values."""
        config = VisionSensorConfig(
            width=1920,
            height=1080,
            fps=60,
            enable_segmentation=True,
        )

        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 60
        assert config.enable_segmentation is True


class TestVisionData:
    """Test VisionData dataclass."""

    def test_vision_data_creation(self):
        """Test VisionData creation."""
        data = VisionData(
            timestamp=time.time(),
            vlm_description="A room with a table",
            detected_objects=[{"label": "table", "confidence": 0.95}],
        )

        assert data.vlm_description == "A room with a table"
        assert len(data.detected_objects) == 1
        assert data.detected_objects[0]["label"] == "table"

    def test_vision_data_defaults(self):
        """Test VisionData default values."""
        data = VisionData(timestamp=time.time())

        assert data.rgb_frame is None
        assert data.depth_frame is None
        assert data.segmentation_mask is None
        assert data.bev_grid is None
        assert data.pointcloud is None
        assert data.vlm_description is None
        assert data.detected_objects == []


class TestVisionSensorInitialization:
    """Test VisionSensor initialization."""

    def test_sensor_initialization(self, config):
        """Test VisionSensor initialization."""
        sensor = VisionSensor(config=config)

        assert sensor.descriptor_for_LLM == "Vision"
        assert isinstance(sensor.message_buffer, Queue)
        assert sensor.latest_description is None
        assert sensor.latest_objects == []

    def test_sensor_config_access(self, config):
        """Test sensor has access to config."""
        sensor = VisionSensor(config=config)

        assert sensor.config == config
        assert sensor.config.vlm_model == "gemini-pro-vision"


class TestVisionSensorVLMCallback:
    """Test VisionSensor VLM callback handling."""

    def test_handle_vlm_result(self, config):
        """Test handling VLM result."""
        sensor = VisionSensor(config=config)
        objects = [{"label": "person", "confidence": 0.9}]

        sensor._handle_vlm_result("A person standing", objects)

        assert sensor.message_buffer.qsize() == 1
        vision_data = sensor.message_buffer.get()
        assert vision_data.vlm_description == "A person standing"
        assert vision_data.detected_objects == objects


class TestVisionSensorPoll:
    """Test VisionSensor polling."""

    @pytest.mark.asyncio
    async def test_poll_with_data(self, config):
        """Test polling when data is available."""
        sensor = VisionSensor(config=config)
        vision_data = VisionData(
            timestamp=time.time(),
            vlm_description="Test scene",
        )
        sensor.message_buffer.put(vision_data)

        result = await sensor._poll()

        assert result is not None
        assert result.vlm_description == "Test scene"

    @pytest.mark.asyncio
    async def test_poll_empty_buffer(self, config):
        """Test polling when buffer is empty."""
        sensor = VisionSensor(config=config)

        result = await sensor._poll()

        assert result is None


class TestVisionSensorRawToText:
    """Test VisionSensor raw to text conversion."""

    @pytest.mark.asyncio
    async def test_raw_to_text_with_description(self, config):
        """Test converting VisionData with description."""
        sensor = VisionSensor(config=config)
        vision_data = VisionData(
            timestamp=time.time(),
            vlm_description="A kitchen with appliances",
        )

        message = await sensor._raw_to_text(vision_data)

        assert message is not None
        assert "Scene: A kitchen with appliances" in message.message

    @pytest.mark.asyncio
    async def test_raw_to_text_with_objects(self, config):
        """Test converting VisionData with detected objects."""
        sensor = VisionSensor(config=config)
        vision_data = VisionData(
            timestamp=time.time(),
            detected_objects=[
                {"label": "chair"},
                {"label": "table"},
            ],
        )

        message = await sensor._raw_to_text(vision_data)

        assert message is not None
        assert "Objects:" in message.message
        assert "chair" in message.message
        assert "table" in message.message

    @pytest.mark.asyncio
    async def test_raw_to_text_none(self, config):
        """Test converting None input."""
        sensor = VisionSensor(config=config)

        message = await sensor._raw_to_text(None)

        assert message is None


class TestVisionSensorFormattedBuffer:
    """Test VisionSensor formatted buffer output."""

    def test_formatted_latest_buffer_empty(self, config):
        """Test formatted buffer when empty."""
        sensor = VisionSensor(config=config)

        result = sensor.formatted_latest_buffer()

        assert result is None

    def test_formatted_latest_buffer_with_description(self, config):
        """Test formatted buffer with description."""
        sensor = VisionSensor(config=config)
        sensor.latest_description = "A living room"

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Vision" in result
        assert "A living room" in result

    def test_formatted_latest_buffer_clears(self, config):
        """Test formatted buffer clears after read."""
        sensor = VisionSensor(config=config)
        sensor.latest_description = "Test"
        sensor.latest_objects = [{"label": "test"}]

        sensor.formatted_latest_buffer()

        assert sensor.latest_description is None
        assert sensor.latest_objects == []


class TestVisionSensorFrameAccess:
    """Test VisionSensor frame access methods."""

    def test_get_current_frame(self, config):
        """Test getting current RGB frame."""
        sensor = VisionSensor(config=config)

        result = sensor.get_current_frame()

        # TODO: Returns None until implemented
        assert result is None

    def test_get_current_depth(self, config):
        """Test getting current depth frame."""
        sensor = VisionSensor(config=config)

        result = sensor.get_current_depth()

        # TODO: Returns None until implemented
        assert result is None


class TestVisionSensorLifecycle:
    """Test VisionSensor start/stop lifecycle."""

    def test_start(self, config, caplog):
        """Test starting the sensor."""
        sensor = VisionSensor(config=config)

        sensor.start()

        assert "started" in caplog.text

    def test_stop(self, config, caplog):
        """Test stopping the sensor."""
        sensor = VisionSensor(config=config)

        sensor.stop()

        assert "stopped" in caplog.text
