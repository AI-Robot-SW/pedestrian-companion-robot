import time
from unittest.mock import patch

import pytest

from providers.bev_occupancy_grid_provider import BEVOccupancyGridProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    BEVOccupancyGridProvider.reset()  # type: ignore
    yield
    BEVOccupancyGridProvider.reset()  # type: ignore


@pytest.fixture
def provider_params():
    return {
        "res": 0.05,
        "width": 50,
        "height": 60,
        "origin_x": 0.0,
        "origin_y": -1.5,
        "dx": -0.34,
        "dy": 0.0,
        "closing_kernel_size": 1,
    }


def test_initialization(provider_params):
    """Test BEVOccupancyGridProvider initialization."""
    provider = BEVOccupancyGridProvider(**provider_params)

    assert provider.res == provider_params["res"]
    assert provider.width == provider_params["width"]
    assert provider.height == provider_params["height"]
    assert provider.origin_x == provider_params["origin_x"]
    assert provider.origin_y == provider_params["origin_y"]
    assert provider.dx == provider_params["dx"]
    assert provider.dy == provider_params["dy"]
    assert provider.closing_kernel_size == provider_params["closing_kernel_size"]
    assert provider._bev_image is None
    assert provider._occupancy_grid is None
    assert provider._data is None
    assert not provider.running
    assert provider._thread is None


def test_singleton_pattern(provider_params):
    """Test that BEVOccupancyGridProvider follows singleton pattern."""
    provider1 = BEVOccupancyGridProvider(**provider_params)
    provider2 = BEVOccupancyGridProvider(**provider_params)

    assert provider1 is provider2


def test_start(provider_params):
    """Test starting the provider."""
    provider = BEVOccupancyGridProvider(**provider_params)
    provider.start()

    assert provider.running
    assert provider._thread is not None
    assert provider._thread.is_alive()

    provider.stop()


def test_stop(provider_params):
    """Test stopping the provider."""
    provider = BEVOccupancyGridProvider(**provider_params)
    provider.start()
    assert provider.running

    provider.stop()

    assert not provider.running
    time.sleep(0.1)
    if provider._thread:
        assert not provider._thread.is_alive()


def test_data_property(provider_params):
    """Test data property access."""
    provider = BEVOccupancyGridProvider(**provider_params)

    assert provider.data is None

    test_data = {
        "bev_image": None,
        "occupancy_grid": None,
        "timestamp": time.time(),
    }
    provider._data = test_data
    assert provider.data == test_data


def test_update_data_placeholder(provider_params):
    """Test _update_data method (placeholder implementation)."""
    provider = BEVOccupancyGridProvider(**provider_params)
    provider._update_data()

    assert provider._data is not None
    assert "bev_image" in provider._data
    assert "occupancy_grid" in provider._data
    assert "timestamp" in provider._data


def test_error_handling_in_run(provider_params):
    """Test error handling in _run method."""
    provider = BEVOccupancyGridProvider(**provider_params)

    with patch.object(provider, "_update_data", side_effect=Exception("Test error")):
        provider.start()
        time.sleep(0.1)
        provider.stop()

    assert not provider.running
