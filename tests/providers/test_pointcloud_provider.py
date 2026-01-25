import time
from unittest.mock import patch

import pytest

from providers.pointcloud_provider import PointCloudProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    PointCloudProvider.reset()  # type: ignore
    yield
    PointCloudProvider.reset()  # type: ignore


@pytest.fixture
def provider_params():
    return {
        "range_max": 3.0,
        "stride": 2,
        "sync_slop": 0.05,
    }


def test_initialization(provider_params):
    """Test PointCloudProvider initialization."""
    provider = PointCloudProvider(**provider_params)

    assert provider.range_max == provider_params["range_max"]
    assert provider.stride == provider_params["stride"]
    assert provider.sync_slop == provider_params["sync_slop"]
    assert provider._pointcloud_data is None
    assert provider._data is None
    assert not provider.running
    assert provider._thread is None


def test_singleton_pattern(provider_params):
    """Test that PointCloudProvider follows singleton pattern."""
    provider1 = PointCloudProvider(**provider_params)
    provider2 = PointCloudProvider(**provider_params)

    assert provider1 is provider2


def test_start(provider_params):
    """Test starting the provider."""
    provider = PointCloudProvider(**provider_params)
    provider.start()

    assert provider.running
    assert provider._thread is not None
    assert provider._thread.is_alive()

    provider.stop()


def test_stop(provider_params):
    """Test stopping the provider."""
    provider = PointCloudProvider(**provider_params)
    provider.start()
    assert provider.running

    provider.stop()

    assert not provider.running
    time.sleep(0.1)
    if provider._thread:
        assert not provider._thread.is_alive()


def test_data_property(provider_params):
    """Test data property access."""
    provider = PointCloudProvider(**provider_params)

    assert provider.data is None

    test_data = {"pointcloud": None, "timestamp": time.time()}
    provider._data = test_data
    assert provider.data == test_data


def test_check_synchronization(provider_params):
    """Test _check_synchronization method (placeholder implementation)."""
    provider = PointCloudProvider(**provider_params)

    result = provider._check_synchronization(None, None, None)
    assert result is False


def test_update_data_placeholder(provider_params):
    """Test _update_data method (placeholder implementation)."""
    provider = PointCloudProvider(**provider_params)
    provider._update_data()

    assert provider._data is not None
    assert "pointcloud" in provider._data
    assert "timestamp" in provider._data


def test_error_handling_in_run(provider_params):
    """Test error handling in _run method."""
    provider = PointCloudProvider(**provider_params)

    with patch.object(provider, "_update_data", side_effect=Exception("Test error")):
        provider.start()
        time.sleep(0.1)
        provider.stop()

    assert not provider.running
