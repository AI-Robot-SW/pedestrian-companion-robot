"""
Tests for TTSProvider.
"""

from unittest.mock import MagicMock, patch

import pytest

from providers.tts_provider import TTSBackend, TTSProvider, TTSState


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    TTSProvider.reset()  # type: ignore
    yield
    TTSProvider.reset()  # type: ignore


@pytest.fixture
def default_config():
    """Default configuration for TTSProvider."""
    return {
        "url": "https://api.openmind.org/api/core/elevenlabs/tts",
        "api_key": "test_api_key",
        "backend_api_key": None,
        "backend": TTSBackend.ELEVENLABS,
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "model_id": "eleven_flash_v2_5",
        "output_format": "mp3_44100_128",
        "language": "ko",
        "enable_tts_interrupt": False,
    }


@pytest.fixture
def custom_config():
    """Custom configuration for TTSProvider."""
    return {
        "url": "https://custom.api.com/tts",
        "api_key": "custom_api_key",
        "backend_api_key": "elevenlabs_key",
        "backend": TTSBackend.ELEVENLABS,
        "voice_id": "custom_voice",
        "model_id": "custom_model",
        "output_format": "pcm_44100",
        "language": "en",
        "enable_tts_interrupt": True,
    }


class TestTTSBackendEnum:
    """Test TTSBackend enum."""

    def test_elevenlabs_value(self):
        """Test ElevenLabs backend value."""
        assert TTSBackend.ELEVENLABS.value == "elevenlabs"

    def test_riva_value(self):
        """Test Riva backend value."""
        assert TTSBackend.RIVA.value == "riva"

    def test_google_value(self):
        """Test Google backend value."""
        assert TTSBackend.GOOGLE.value == "google"

    def test_local_value(self):
        """Test Local backend value."""
        assert TTSBackend.LOCAL.value == "local"


class TestTTSStateEnum:
    """Test TTSState enum."""

    def test_idle_value(self):
        """Test idle state value."""
        assert TTSState.IDLE.value == "idle"

    def test_processing_value(self):
        """Test processing state value."""
        assert TTSState.PROCESSING.value == "processing"

    def test_speaking_value(self):
        """Test speaking state value."""
        assert TTSState.SPEAKING.value == "speaking"

    def test_error_value(self):
        """Test error state value."""
        assert TTSState.ERROR.value == "error"


class TestTTSProviderInitialization:
    """Test TTSProvider initialization."""

    def test_default_initialization(self, default_config):
        """Test TTSProvider initialization with default values."""
        provider = TTSProvider(**default_config)

        assert provider.url == "https://api.openmind.org/api/core/elevenlabs/tts"
        assert provider.api_key == "test_api_key"
        assert provider.backend == TTSBackend.ELEVENLABS
        assert provider.voice_id == "JBFqnCBsd6RMkjVDRZzb"
        assert provider.model_id == "eleven_flash_v2_5"
        assert provider._output_format == "mp3_44100_128"
        assert provider._language == "ko"
        assert not provider.running

    def test_custom_initialization(self, custom_config):
        """Test TTSProvider initialization with custom values."""
        provider = TTSProvider(**custom_config)

        assert provider.url == "https://custom.api.com/tts"
        assert provider.backend_api_key == "elevenlabs_key"
        assert provider.voice_id == "custom_voice"
        assert provider.model_id == "custom_model"
        assert provider._enable_tts_interrupt is True

    def test_singleton_pattern(self, default_config):
        """Test that TTSProvider follows singleton pattern."""
        provider1 = TTSProvider(**default_config)
        provider2 = TTSProvider(**default_config)

        assert provider1 is provider2


class TestTTSProviderProperties:
    """Test TTSProvider properties."""

    def test_voice_id_getter(self, default_config):
        """Test voice_id getter."""
        provider = TTSProvider(**default_config)

        assert provider.voice_id == "JBFqnCBsd6RMkjVDRZzb"

    def test_voice_id_setter(self, default_config):
        """Test voice_id setter."""
        provider = TTSProvider(**default_config)

        provider.voice_id = "new_voice"

        assert provider.voice_id == "new_voice"

    def test_model_id_getter(self, default_config):
        """Test model_id getter."""
        provider = TTSProvider(**default_config)

        assert provider.model_id == "eleven_flash_v2_5"

    def test_model_id_setter(self, default_config):
        """Test model_id setter."""
        provider = TTSProvider(**default_config)

        provider.model_id = "new_model"

        assert provider.model_id == "new_model"


class TestTTSProviderCallbacks:
    """Test TTSProvider callback registration."""

    def test_register_tts_state_callback(self, default_config):
        """Test registering TTS state callback."""
        provider = TTSProvider(**default_config)
        callback = MagicMock()

        provider.register_tts_state_callback(callback)

        assert callback in provider._state_callbacks

    def test_unregister_tts_state_callback(self, default_config):
        """Test unregistering TTS state callback."""
        provider = TTSProvider(**default_config)
        callback = MagicMock()

        provider.register_tts_state_callback(callback)
        provider.unregister_tts_state_callback(callback)

        assert callback not in provider._state_callbacks

    def test_register_audio_callback(self, default_config):
        """Test registering audio callback."""
        provider = TTSProvider(**default_config)
        callback = MagicMock()

        provider.register_audio_callback(callback)

        assert callback in provider._audio_callbacks

    def test_unregister_audio_callback(self, default_config):
        """Test unregistering audio callback."""
        provider = TTSProvider(**default_config)
        callback = MagicMock()

        provider.register_audio_callback(callback)
        provider.unregister_audio_callback(callback)

        assert callback not in provider._audio_callbacks


class TestTTSProviderMessage:
    """Test TTSProvider message creation."""

    def test_create_pending_message(self, default_config):
        """Test creating pending message."""
        provider = TTSProvider(**default_config)

        message = provider.create_pending_message("안녕하세요")

        assert message["text"] == "안녕하세요"
        assert message["voice_id"] == "JBFqnCBsd6RMkjVDRZzb"
        assert message["model_id"] == "eleven_flash_v2_5"
        assert message["output_format"] == "mp3_44100_128"
        assert message["language"] == "ko"

    def test_create_pending_message_with_backend_key(self, custom_config):
        """Test creating pending message includes backend API key."""
        provider = TTSProvider(**custom_config)

        message = provider.create_pending_message("Hello")

        assert message["backend_api_key"] == "elevenlabs_key"

    def test_add_pending_message_not_running(self, default_config, caplog):
        """Test adding message when not running logs warning."""
        provider = TTSProvider(**default_config)

        provider.add_pending_message("test")

        assert "not running" in caplog.text

    def test_add_pending_message_string(self, default_config):
        """Test adding string message."""
        provider = TTSProvider(**default_config)
        provider.start()

        provider.add_pending_message("안녕하세요")

        assert provider.get_pending_message_count() == 1

        provider.stop()

    def test_add_pending_message_dict(self, default_config):
        """Test adding dict message."""
        provider = TTSProvider(**default_config)
        provider.start()

        message = {"text": "test", "voice_id": "custom"}
        provider.add_pending_message(message)

        assert provider.get_pending_message_count() == 1

        provider.stop()


class TestTTSProviderState:
    """Test TTSProvider state methods."""

    def test_get_current_state_initial(self, default_config):
        """Test initial state is IDLE."""
        provider = TTSProvider(**default_config)

        assert provider.get_current_state() == TTSState.IDLE

    def test_get_pending_message_count_empty(self, default_config):
        """Test pending message count when empty."""
        provider = TTSProvider(**default_config)

        assert provider.get_pending_message_count() == 0


class TestTTSProviderLifecycle:
    """Test TTSProvider start/stop lifecycle."""

    def test_start(self, default_config):
        """Test starting the provider."""
        provider = TTSProvider(**default_config)
        provider.start()

        assert provider.running is True

        provider.stop()

    def test_start_already_running(self, default_config, caplog):
        """Test starting already running provider logs warning."""
        provider = TTSProvider(**default_config)
        provider.start()
        provider.start()

        assert "already running" in caplog.text

        provider.stop()

    def test_stop(self, default_config):
        """Test stopping the provider."""
        provider = TTSProvider(**default_config)
        provider.start()
        provider.stop()

        assert provider.running is False

    def test_stop_not_running(self, default_config, caplog):
        """Test stopping not running provider logs warning."""
        provider = TTSProvider(**default_config)
        provider.stop()

        assert "not running" in caplog.text


class TestTTSProviderInterrupt:
    """Test TTSProvider interrupt functionality."""

    def test_interrupt_not_enabled(self, default_config, caplog):
        """Test interrupt when not enabled logs warning."""
        provider = TTSProvider(**default_config)

        provider.interrupt()

        assert "not enabled" in caplog.text

    def test_interrupt_clears_queue(self, custom_config):
        """Test interrupt clears pending queue."""
        provider = TTSProvider(**custom_config)
        provider.start()
        provider.add_pending_message("test1")
        provider.add_pending_message("test2")

        provider.interrupt()

        assert provider.get_pending_message_count() == 0
        assert provider.get_current_state() == TTSState.IDLE

        provider.stop()


class TestTTSProviderConfiguration:
    """Test TTSProvider configuration methods."""

    def test_configure_voice_id(self, default_config):
        """Test configuring voice ID."""
        provider = TTSProvider(**default_config)

        provider.configure(voice_id="new_voice")

        assert provider.voice_id == "new_voice"

    def test_configure_model_id(self, default_config):
        """Test configuring model ID."""
        provider = TTSProvider(**default_config)

        provider.configure(model_id="new_model")

        assert provider.model_id == "new_model"

    def test_configure_language(self, default_config):
        """Test configuring language."""
        provider = TTSProvider(**default_config)

        provider.configure(language="en")

        assert provider._language == "en"

    def test_configure_output_format(self, default_config):
        """Test configuring output format."""
        provider = TTSProvider(**default_config)

        provider.configure(output_format="pcm_44100")

        assert provider._output_format == "pcm_44100"
