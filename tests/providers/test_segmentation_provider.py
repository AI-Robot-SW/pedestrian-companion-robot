import time
from unittest.mock import patch

import pytest

from providers.segmentation_provider import SegmentationProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    SegmentationProvider.reset()  # type: ignore
    yield
    SegmentationProvider.reset()  # type: ignore


@pytest.fixture
def engine_path():
    return "/path/to/engine.engine"


def test_initialization(engine_path):
    """Test SegmentationProvider initialization."""
    provider = SegmentationProvider(engine_path=engine_path)

    assert provider.engine_path == engine_path
    assert provider._segmented_image is None
    assert provider._classes is None
    assert provider._data is None
    assert not provider.running
    assert provider._thread is None


def test_singleton_pattern(engine_path):
    """Test that SegmentationProvider follows singleton pattern."""
    provider1 = SegmentationProvider(engine_path=engine_path)
    provider2 = SegmentationProvider(engine_path=engine_path)

    assert provider1 is provider2


def test_start(engine_path):
    """Test starting the provider."""
    provider = SegmentationProvider(engine_path=engine_path)
    provider.start()

    assert provider.running
    assert provider._thread is not None
    assert provider._thread.is_alive()

    # Cleanup
    provider.stop()


def test_stop(engine_path):
    """Test stopping the provider."""
    provider = SegmentationProvider(engine_path=engine_path)
    provider.start()
    assert provider.running

    provider.stop()

    assert not provider.running
    time.sleep(0.1)
    if provider._thread:
        assert not provider._thread.is_alive()


def test_data_property(engine_path):
    """Test data property access."""
    provider = SegmentationProvider(engine_path=engine_path)

    assert provider.data is None

    test_data = {
        "segmented_image": None,
        "classes": None,
        "timestamp": time.time(),
    }
    provider._data = test_data
    assert provider.data == test_data


def test_update_data_placeholder(engine_path):
    """Test _update_data method (placeholder implementation)."""
    provider = SegmentationProvider(engine_path=engine_path)
    provider._update_data()

    assert provider._data is not None
    assert "segmented_image" in provider._data
    assert "classes" in provider._data
    assert "timestamp" in provider._data


def test_error_handling_in_run(engine_path):
    """Test error handling in _run method."""
    provider = SegmentationProvider(engine_path=engine_path)

    with patch.object(provider, "_update_data", side_effect=Exception("Test error")):
        provider.start()
        time.sleep(0.1)
        provider.stop()

    assert not provider.running
