import time
from unittest.mock import patch

import pytest

from providers.realsense_camera_provider import RealSenseCameraProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    RealSenseCameraProvider.reset()  # type: ignore
    yield
    RealSenseCameraProvider.reset()  # type: ignore


@pytest.fixture
def camera_index():
    return 0


def test_initialization(camera_index):
    """Test RealSenseCameraProvider initialization."""
    provider = RealSenseCameraProvider(camera_index=camera_index)

    assert provider.camera_index == camera_index
    assert provider._rgb_data is None
    assert provider._depth_data is None
    assert provider._camera_info is None
    assert provider._data is None
    assert not provider.running
    assert provider._thread is None


def test_singleton_pattern(camera_index):
    """Test that RealSenseCameraProvider follows singleton pattern."""
    provider1 = RealSenseCameraProvider(camera_index=camera_index)
    provider2 = RealSenseCameraProvider(camera_index=camera_index)

    assert provider1 is provider2


def test_start(camera_index):
    """Test starting the provider."""
    provider = RealSenseCameraProvider(camera_index=camera_index)
    provider.start()

    assert provider.running
    assert provider._thread is not None
    assert provider._thread.is_alive()

    # Cleanup
    provider.stop()


def test_stop(camera_index):
    """Test stopping the provider."""
    provider = RealSenseCameraProvider(camera_index=camera_index)
    provider.start()
    assert provider.running

    provider.stop()

    assert not provider.running
    # Thread should be stopped (may take a moment)
    time.sleep(0.1)
    if provider._thread:
        assert not provider._thread.is_alive()


def test_data_property(camera_index):
    """Test data property access."""
    provider = RealSenseCameraProvider(camera_index=camera_index)

    # Initially should be None
    assert provider.data is None

    # After setting _data, should return it
    test_data = {"rgb": None, "depth": None, "camera_info": None, "timestamp": time.time()}
    provider._data = test_data
    assert provider.data == test_data


def test_update_data_placeholder(camera_index):
    """Test _update_data method (placeholder implementation)."""
    provider = RealSenseCameraProvider(camera_index=camera_index)
    provider._update_data()

    # Should update _data with structure
    assert provider._data is not None
    assert "rgb" in provider._data
    assert "depth" in provider._data
    assert "camera_info" in provider._data
    assert "timestamp" in provider._data


def test_error_handling_in_run(camera_index):
    """Test error handling in _run method."""
    provider = RealSenseCameraProvider(camera_index=camera_index)

    # Simulate error in _update_data
    with patch.object(provider, "_update_data", side_effect=Exception("Test error")):
        provider.start()
        time.sleep(0.1)  # Let thread run briefly
        provider.stop()

    # Should handle error gracefully
    assert not provider.running
