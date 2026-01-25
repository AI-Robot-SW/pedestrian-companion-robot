"""
Tests for SpeakerProvider.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from providers.speaker_provider import SpeakerProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    SpeakerProvider.reset()  # type: ignore
    yield
    SpeakerProvider.reset()  # type: ignore


@pytest.fixture
def default_config():
    """Default configuration for SpeakerProvider."""
    return {
        "sample_rate": 44100,
        "channels": 1,
        "device_id": None,
        "device_name": None,
        "volume": 1.0,
        "buffer_size": 4096,
    }


@pytest.fixture
def custom_config():
    """Custom configuration for SpeakerProvider."""
    return {
        "sample_rate": 48000,
        "channels": 2,
        "device_id": 1,
        "device_name": "Test Speaker",
        "volume": 0.5,
        "buffer_size": 8192,
    }


class TestSpeakerProviderInitialization:
    """Test SpeakerProvider initialization."""

    def test_default_initialization(self, default_config):
        """Test SpeakerProvider initialization with default values."""
        provider = SpeakerProvider(**default_config)

        assert provider.sample_rate == 44100
        assert provider.channels == 1
        assert provider.device_id is None
        assert provider.device_name is None
        assert provider.volume == 1.0
        assert provider.buffer_size == 4096
        assert not provider.running

    def test_custom_initialization(self, custom_config):
        """Test SpeakerProvider initialization with custom values."""
        provider = SpeakerProvider(**custom_config)

        assert provider.sample_rate == 48000
        assert provider.channels == 2
        assert provider.device_id == 1
        assert provider.device_name == "Test Speaker"
        assert provider.volume == 0.5
        assert provider.buffer_size == 8192

    def test_singleton_pattern(self, default_config):
        """Test that SpeakerProvider follows singleton pattern."""
        provider1 = SpeakerProvider(**default_config)
        provider2 = SpeakerProvider(**default_config)

        assert provider1 is provider2

    def test_volume_clamping_high(self):
        """Test volume is clamped to maximum 1.0."""
        provider = SpeakerProvider(volume=1.5)

        assert provider.volume == 1.0

    def test_volume_clamping_low(self):
        """Test volume is clamped to minimum 0.0."""
        provider = SpeakerProvider(volume=-0.5)

        assert provider.volume == 0.0


class TestSpeakerProviderVolume:
    """Test SpeakerProvider volume control."""

    def test_set_volume(self, default_config):
        """Test setting volume."""
        provider = SpeakerProvider(**default_config)

        provider.volume = 0.7

        assert provider.volume == 0.7

    def test_set_volume_clamping_high(self, default_config):
        """Test volume setter clamps high values."""
        provider = SpeakerProvider(**default_config)

        provider.volume = 2.0

        assert provider.volume == 1.0

    def test_set_volume_clamping_low(self, default_config):
        """Test volume setter clamps low values."""
        provider = SpeakerProvider(**default_config)

        provider.volume = -1.0

        assert provider.volume == 0.0


class TestSpeakerProviderCallbacks:
    """Test SpeakerProvider callback registration."""

    def test_register_playback_callback(self, default_config):
        """Test registering playback callback."""
        provider = SpeakerProvider(**default_config)
        callback = MagicMock()

        provider.register_playback_callback(callback)

        assert callback in provider._playback_callbacks

    def test_unregister_playback_callback(self, default_config):
        """Test unregistering playback callback."""
        provider = SpeakerProvider(**default_config)
        callback = MagicMock()

        provider.register_playback_callback(callback)
        provider.unregister_playback_callback(callback)

        assert callback not in provider._playback_callbacks

    def test_duplicate_callback_registration(self, default_config):
        """Test that duplicate callbacks are not registered."""
        provider = SpeakerProvider(**default_config)
        callback = MagicMock()

        provider.register_playback_callback(callback)
        provider.register_playback_callback(callback)

        assert provider._playback_callbacks.count(callback) == 1


class TestSpeakerProviderQueue:
    """Test SpeakerProvider queue management."""

    def test_queue_audio_not_running(self, default_config, caplog):
        """Test queueing audio when not running logs warning."""
        provider = SpeakerProvider(**default_config)

        provider.queue_audio(b"audio_data")

        assert "not running" in caplog.text

    def test_queue_audio_running(self, default_config):
        """Test queueing audio when running."""
        provider = SpeakerProvider(**default_config)
        provider.start()

        provider.queue_audio(b"audio_data")

        assert provider.get_queue_size() == 1

        provider.stop()

    def test_clear_queue(self, default_config):
        """Test clearing the audio queue."""
        provider = SpeakerProvider(**default_config)
        provider.start()
        provider.queue_audio(b"audio_data_1")
        provider.queue_audio(b"audio_data_2")

        provider.clear_queue()

        assert provider.get_queue_size() == 0

        provider.stop()

    def test_get_queue_size_empty(self, default_config):
        """Test queue size when empty."""
        provider = SpeakerProvider(**default_config)

        assert provider.get_queue_size() == 0


class TestSpeakerProviderState:
    """Test SpeakerProvider state methods."""

    def test_is_playing_initial(self, default_config):
        """Test initial playing state is False."""
        provider = SpeakerProvider(**default_config)

        assert provider.is_playing() is False


class TestSpeakerProviderLifecycle:
    """Test SpeakerProvider start/stop lifecycle."""

    def test_start(self, default_config):
        """Test starting the provider."""
        provider = SpeakerProvider(**default_config)
        provider.start()

        assert provider.running is True

        provider.stop()

    def test_start_already_running(self, default_config, caplog):
        """Test starting already running provider logs warning."""
        provider = SpeakerProvider(**default_config)
        provider.start()
        provider.start()

        assert "already running" in caplog.text

        provider.stop()

    def test_stop(self, default_config):
        """Test stopping the provider."""
        provider = SpeakerProvider(**default_config)
        provider.start()
        provider.stop()

        assert provider.running is False

    def test_stop_not_running(self, default_config, caplog):
        """Test stopping not running provider logs warning."""
        provider = SpeakerProvider(**default_config)
        provider.stop()

        assert "not running" in caplog.text

    def test_stop_playback(self, default_config):
        """Test stopping playback."""
        provider = SpeakerProvider(**default_config)
        provider.start()
        provider.queue_audio(b"audio_data")

        provider.stop_playback()

        assert provider.get_queue_size() == 0

        provider.stop()


class TestSpeakerProviderConfiguration:
    """Test SpeakerProvider configuration methods."""

    def test_configure_sample_rate(self, default_config):
        """Test configuring sample rate."""
        provider = SpeakerProvider(**default_config)

        provider.configure(sample_rate=48000)

        assert provider.sample_rate == 48000

    def test_configure_channels(self, default_config):
        """Test configuring channels."""
        provider = SpeakerProvider(**default_config)

        provider.configure(channels=2)

        assert provider.channels == 2

    def test_configure_volume(self, default_config):
        """Test configuring volume."""
        provider = SpeakerProvider(**default_config)

        provider.configure(volume=0.5)

        assert provider.volume == 0.5

    def test_configure_device_id(self, default_config):
        """Test configuring device ID."""
        provider = SpeakerProvider(**default_config)

        provider.configure(device_id=2)

        assert provider.device_id == 2
