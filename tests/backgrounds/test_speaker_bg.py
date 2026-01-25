"""
Tests for SpeakerBg.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.speaker_bg import SpeakerBg, SpeakerBgConfig


@pytest.fixture
def config():
    """Configuration for SpeakerBg."""
    return SpeakerBgConfig(
        sample_rate=44100,
        channels=1,
        device_id=None,
        device_name=None,
        volume=1.0,
        buffer_size=4096,
        health_check_interval_sec=10.0,
    )


@pytest.fixture
def config_custom():
    """Custom configuration for SpeakerBg."""
    return SpeakerBgConfig(
        sample_rate=48000,
        channels=2,
        device_id=1,
        device_name="Test Speaker",
        volume=0.5,
        buffer_size=8192,
        health_check_interval_sec=5.0,
    )


@pytest.fixture
def config_default():
    """Default configuration for SpeakerBg."""
    return SpeakerBgConfig()


class TestSpeakerBgConfig:
    """Test SpeakerBgConfig initialization."""

    def test_config_default_values(self):
        """Test SpeakerBgConfig default values."""
        config = SpeakerBgConfig()

        assert config.sample_rate == 44100
        assert config.channels == 1
        assert config.device_id is None
        assert config.device_name is None
        assert config.volume == 1.0
        assert config.buffer_size == 4096
        assert config.health_check_interval_sec == 10.0

    def test_config_custom_values(self):
        """Test SpeakerBgConfig with custom values."""
        config = SpeakerBgConfig(
            sample_rate=48000,
            channels=2,
            device_id=1,
            volume=0.5,
            buffer_size=8192,
        )

        assert config.sample_rate == 48000
        assert config.channels == 2
        assert config.device_id == 1
        assert config.volume == 0.5
        assert config.buffer_size == 8192


class TestSpeakerBgInitialization:
    """Test SpeakerBg initialization."""

    @patch("backgrounds.plugins.speaker_bg.SpeakerProvider")
    def test_background_initialization(self, mock_provider_class, config):
        """Test SpeakerBg initialization."""
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        background = SpeakerBg(config=config)

        mock_provider_class.assert_called_once_with(
            sample_rate=44100,
            channels=1,
            device_id=None,
            device_name=None,
            volume=1.0,
            buffer_size=4096,
        )
        assert background.speaker_provider is mock_provider_instance
        mock_provider_instance.start.assert_called_once()

    @patch("backgrounds.plugins.speaker_bg.SpeakerProvider")
    def test_background_initialization_custom(self, mock_provider_class, config_custom):
        """Test SpeakerBg initialization with custom config."""
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        background = SpeakerBg(config=config_custom)

        mock_provider_class.assert_called_once_with(
            sample_rate=48000,
            channels=2,
            device_id=1,
            device_name="Test Speaker",
            volume=0.5,
            buffer_size=8192,
        )

    def test_background_name(self, config):
        """Test that background has correct name."""
        background = SpeakerBg(config=config)

        assert background.name == "SpeakerBg"

    def test_background_config_access(self, config):
        """Test that background has access to config."""
        background = SpeakerBg(config=config)

        assert background.config == config
        assert background.config.sample_rate == 44100
        assert background.config.volume == 1.0


class TestSpeakerBgHealthCheck:
    """Test SpeakerBg health check functionality."""

    def test_health_check_returns_true(self, config):
        """Test health check returns True when provider is healthy."""
        background = SpeakerBg(config=config)

        result = background._health_check()

        assert result is True

    def test_consecutive_failures_tracking(self, config):
        """Test consecutive failures are tracked."""
        background = SpeakerBg(config=config)

        assert background._consecutive_failures == 0
        assert background._max_failures == 3


class TestSpeakerBgRun:
    """Test SpeakerBg run method."""

    def test_run_respects_health_check_interval(self, config):
        """Test run method respects health check interval."""
        background = SpeakerBg(config=config)
        background._last_health_check = time.time()

        start_time = time.time()
        background.run()
        elapsed = time.time() - start_time

        assert elapsed >= 0.9

    def test_run_performs_health_check_after_interval(self, config):
        """Test run method performs health check after interval."""
        background = SpeakerBg(config=config)
        background._last_health_check = time.time() - 15.0

        background.run()

        assert time.time() - background._last_health_check < 2.0

    def test_run_resets_failures_on_success(self, config):
        """Test run resets failure counter on successful health check."""
        background = SpeakerBg(config=config)
        background._consecutive_failures = 2
        background._last_health_check = time.time() - 15.0

        background.run()

        assert background._consecutive_failures == 0
