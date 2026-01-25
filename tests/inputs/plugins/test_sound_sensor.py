"""
Tests for SoundSensor.
"""

import asyncio
import time
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.sound_sensor import SoundSensor, SoundSensorConfig


@pytest.fixture
def config():
    """Configuration for SoundSensor."""
    return SoundSensorConfig(
        sample_rate=16000,
        chunk_size=1024,
        channels=1,
        device_id=None,
        device_name=None,
        language="ko-KR",
        vad_enabled=True,
        vad_threshold=0.5,
        buffer_duration_ms=200,
    )


@pytest.fixture
def config_custom():
    """Custom configuration for SoundSensor."""
    return SoundSensorConfig(
        sample_rate=48000,
        chunk_size=2048,
        channels=2,
        device_id=1,
        device_name="Test Microphone",
        language="en-US",
        vad_enabled=False,
        vad_threshold=0.7,
        buffer_duration_ms=300,
    )


class TestSoundSensorConfig:
    """Test SoundSensorConfig initialization."""

    def test_config_default_values(self):
        """Test SoundSensorConfig default values."""
        config = SoundSensorConfig()

        assert config.sample_rate == 16000
        assert config.chunk_size == 1024
        assert config.channels == 1
        assert config.device_id is None
        assert config.device_name is None
        assert config.language == "ko-KR"
        assert config.vad_enabled is True
        assert config.vad_threshold == 0.5
        assert config.buffer_duration_ms == 200

    def test_config_custom_values(self):
        """Test SoundSensorConfig with custom values."""
        config = SoundSensorConfig(
            sample_rate=48000,
            language="en-US",
            vad_enabled=False,
        )

        assert config.sample_rate == 48000
        assert config.language == "en-US"
        assert config.vad_enabled is False


class TestSoundSensorInitialization:
    """Test SoundSensor initialization."""

    def test_sensor_initialization(self, config):
        """Test SoundSensor initialization."""
        sensor = SoundSensor(config=config)

        assert sensor.descriptor_for_LLM == "Voice"
        assert isinstance(sensor.message_buffer, Queue)
        assert sensor.messages == []

    def test_sensor_config_access(self, config):
        """Test sensor has access to config."""
        sensor = SoundSensor(config=config)

        assert sensor.config == config
        assert sensor.config.language == "ko-KR"


class TestSoundSensorSTTCallback:
    """Test SoundSensor STT callback handling."""

    def test_handle_stt_result_valid(self, config):
        """Test handling valid STT result."""
        sensor = SoundSensor(config=config)

        sensor._handle_stt_result("안녕하세요")

        assert sensor.message_buffer.qsize() == 1
        assert sensor.message_buffer.get() == "안녕하세요"

    def test_handle_stt_result_empty(self, config):
        """Test handling empty STT result."""
        sensor = SoundSensor(config=config)

        sensor._handle_stt_result("")

        assert sensor.message_buffer.qsize() == 0

    def test_handle_stt_result_whitespace(self, config):
        """Test handling whitespace-only STT result."""
        sensor = SoundSensor(config=config)

        sensor._handle_stt_result("   ")

        assert sensor.message_buffer.qsize() == 0


class TestSoundSensorPoll:
    """Test SoundSensor polling."""

    @pytest.mark.asyncio
    async def test_poll_with_message(self, config):
        """Test polling when message is available."""
        sensor = SoundSensor(config=config)
        sensor.message_buffer.put("테스트 메시지")

        result = await sensor._poll()

        assert result == "테스트 메시지"

    @pytest.mark.asyncio
    async def test_poll_empty_buffer(self, config):
        """Test polling when buffer is empty."""
        sensor = SoundSensor(config=config)

        result = await sensor._poll()

        assert result is None


class TestSoundSensorRawToText:
    """Test SoundSensor raw to text conversion."""

    @pytest.mark.asyncio
    async def test_raw_to_text_valid(self, config):
        """Test converting valid raw input to text."""
        sensor = SoundSensor(config=config)

        message = await sensor._raw_to_text("테스트")

        assert message is not None
        assert message.message == "테스트"
        assert message.timestamp > 0

    @pytest.mark.asyncio
    async def test_raw_to_text_none(self, config):
        """Test converting None input."""
        sensor = SoundSensor(config=config)

        message = await sensor._raw_to_text(None)

        assert message is None

    @pytest.mark.asyncio
    async def test_raw_to_text_accumulates(self, config):
        """Test that raw_to_text accumulates messages."""
        sensor = SoundSensor(config=config)

        await sensor.raw_to_text("첫번째")
        await sensor.raw_to_text("두번째")

        assert len(sensor.messages) == 1
        assert "첫번째" in sensor.messages[0]
        assert "두번째" in sensor.messages[0]


class TestSoundSensorFormattedBuffer:
    """Test SoundSensor formatted buffer output."""

    def test_formatted_latest_buffer_empty(self, config):
        """Test formatted buffer when empty."""
        sensor = SoundSensor(config=config)

        result = sensor.formatted_latest_buffer()

        assert result is None

    def test_formatted_latest_buffer_with_message(self, config):
        """Test formatted buffer with message."""
        sensor = SoundSensor(config=config)
        sensor.messages = ["테스트 메시지"]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Voice" in result
        assert "테스트 메시지" in result
        assert sensor.messages == []  # Buffer should be cleared

    def test_formatted_latest_buffer_format(self, config):
        """Test formatted buffer has correct format."""
        sensor = SoundSensor(config=config)
        sensor.messages = ["안녕하세요"]

        result = sensor.formatted_latest_buffer()

        assert "// START" in result
        assert "// END" in result


class TestSoundSensorLifecycle:
    """Test SoundSensor start/stop lifecycle."""

    def test_start(self, config, caplog):
        """Test starting the sensor."""
        sensor = SoundSensor(config=config)

        sensor.start()

        assert "started" in caplog.text

    def test_stop(self, config, caplog):
        """Test stopping the sensor."""
        sensor = SoundSensor(config=config)

        sensor.stop()

        assert "stopped" in caplog.text
