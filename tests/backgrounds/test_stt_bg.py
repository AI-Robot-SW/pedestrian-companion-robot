"""
Tests for STTBg.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.stt_bg import STTBg, STTBgConfig


@pytest.fixture
def config():
    """Configuration for STTBg."""
    return STTBgConfig(
        api_key="test_api_key",
        base_url=None,
        language="korean",
        enable_interim_results=True,
        sample_rate=16000,
        reconnect_delay_sec=5.0,
        max_reconnect_attempts=3,
        health_check_interval_sec=10.0,
    )


@pytest.fixture
def config_custom():
    """Custom configuration for STTBg."""
    return STTBgConfig(
        api_key="custom_api_key",
        base_url="wss://custom.api.com/asr",
        language="english",
        enable_interim_results=False,
        sample_rate=48000,
        reconnect_delay_sec=10.0,
        max_reconnect_attempts=5,
        health_check_interval_sec=5.0,
    )


@pytest.fixture
def config_default():
    """Default configuration for STTBg."""
    return STTBgConfig()


class TestSTTBgConfig:
    """Test STTBgConfig initialization."""

    def test_config_default_values(self):
        """Test STTBgConfig default values."""
        config = STTBgConfig()

        assert config.api_key is None
        assert config.base_url is None
        assert config.language == "korean"
        assert config.enable_interim_results is True
        assert config.sample_rate == 16000
        assert config.reconnect_delay_sec == 5.0
        assert config.max_reconnect_attempts == 3
        assert config.health_check_interval_sec == 10.0

    def test_config_custom_values(self):
        """Test STTBgConfig with custom values."""
        config = STTBgConfig(
            api_key="test_key",
            language="english",
            enable_interim_results=False,
            max_reconnect_attempts=5,
        )

        assert config.api_key == "test_key"
        assert config.language == "english"
        assert config.enable_interim_results is False
        assert config.max_reconnect_attempts == 5


class TestSTTBgInitialization:
    """Test STTBg initialization."""

    @patch("backgrounds.plugins.stt_bg.AudioProvider")
    @patch("backgrounds.plugins.stt_bg.STTProvider")
    def test_background_initialization(
        self, mock_stt_provider_class, mock_audio_provider_class, config
    ):
        """Test STTBg initialization."""
        mock_stt_instance = MagicMock()
        mock_stt_provider_class.return_value = mock_stt_instance
        mock_audio_instance = MagicMock()
        mock_audio_provider_class.return_value = mock_audio_instance

        background = STTBg(config=config)

        # Verify STT provider was initialized
        assert mock_stt_provider_class.called
        assert background.stt_provider is mock_stt_instance
        mock_stt_instance.start.assert_called_once()

        # Verify audio callback was registered
        mock_audio_instance.register_audio_callback.assert_called_once_with(
            mock_stt_instance.send_audio
        )

    def test_background_name(self, config):
        """Test that background has correct name."""
        background = STTBg(config=config)

        assert background.name == "STTBg"

    def test_background_config_access(self, config):
        """Test that background has access to config."""
        background = STTBg(config=config)

        assert background.config == config
        assert background.config.language == "korean"
        assert background.config.enable_interim_results is True


class TestSTTBgHealthCheck:
    """Test STTBg health check functionality."""

    def test_health_check_returns_true(self, config):
        """Test health check returns True when provider is healthy."""
        background = STTBg(config=config)

        result = background._health_check()

        assert result is True

    def test_consecutive_failures_tracking(self, config):
        """Test consecutive failures are tracked."""
        background = STTBg(config=config)

        assert background._consecutive_failures == 0
        assert background._reconnect_attempts == 0


class TestSTTBgReconnect:
    """Test STTBg reconnect functionality."""

    def test_reconnect_increments_attempts(self, config):
        """Test reconnect increments attempt counter."""
        background = STTBg(config=config)

        background._reconnect()

        assert background._reconnect_attempts == 1

    def test_reconnect_max_attempts(self, config):
        """Test reconnect fails after max attempts."""
        background = STTBg(config=config)
        background._reconnect_attempts = config.max_reconnect_attempts

        result = background._reconnect()

        assert result is False


class TestSTTBgRun:
    """Test STTBg run method."""

    def test_run_respects_health_check_interval(self, config):
        """Test run method respects health check interval."""
        background = STTBg(config=config)
        background._last_health_check = time.time()

        start_time = time.time()
        background.run()
        elapsed = time.time() - start_time

        assert elapsed >= 0.9

    def test_run_performs_health_check_after_interval(self, config):
        """Test run method performs health check after interval."""
        background = STTBg(config=config)
        background._last_health_check = time.time() - 15.0

        background.run()

        assert time.time() - background._last_health_check < 2.0

    def test_run_resets_failures_on_success(self, config):
        """Test run resets failure counters on successful health check."""
        background = STTBg(config=config)
        background._consecutive_failures = 2
        background._reconnect_attempts = 1
        background._last_health_check = time.time() - 15.0

        background.run()

        assert background._consecutive_failures == 0
        assert background._reconnect_attempts == 0
