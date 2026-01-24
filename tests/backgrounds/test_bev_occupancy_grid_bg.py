from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.bev_occupancy_grid_bg import (
    BEVOccupancyGridBg,
    BEVOccupancyGridConfig,
)


@pytest.fixture
def config():
    return BEVOccupancyGridConfig(
        res=0.05,
        width=50,
        height=60,
        origin_x=0.0,
        origin_y=-1.5,
        dx=-0.34,
        dy=0.0,
        closing_kernel_size=1,
    )


@pytest.fixture
def config_default():
    return BEVOccupancyGridConfig()


def test_config_initialization():
    """Test BEVOccupancyGridConfig initialization."""
    config = BEVOccupancyGridConfig()
    assert config.res == 0.05
    assert config.width == 50
    assert config.height == 60
    assert config.origin_x == 0.0
    assert config.origin_y == -1.5
    assert config.dx == -0.34
    assert config.dy == 0.0
    assert config.closing_kernel_size == 1

    config_custom = BEVOccupancyGridConfig(
        res=0.1, width=100, height=120, origin_x=1.0, origin_y=-2.0, dx=-0.5, dy=0.1, closing_kernel_size=2
    )
    assert config_custom.res == 0.1
    assert config_custom.width == 100
    assert config_custom.height == 120


@patch("backgrounds.plugins.bev_occupancy_grid_bg.BEVOccupancyGridProvider")
def test_background_initialization(mock_provider_class, config):
    """Test BEVOccupancyGridBg initialization."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = BEVOccupancyGridBg(config=config)

    mock_provider_class.assert_called_once_with(
        res=0.05,
        width=50,
        height=60,
        origin_x=0.0,
        origin_y=-1.5,
        dx=-0.34,
        dy=0.0,
        closing_kernel_size=1,
    )
    assert background.bev_occupancy_grid_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.bev_occupancy_grid_bg.BEVOccupancyGridProvider")
def test_background_initialization_default(mock_provider_class, config_default):
    """Test BEVOccupancyGridBg initialization with default parameters."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = BEVOccupancyGridBg(config=config_default)

    mock_provider_class.assert_called_once_with(
        res=0.05,
        width=50,
        height=60,
        origin_x=0.0,
        origin_y=-1.5,
        dx=-0.34,
        dy=0.0,
        closing_kernel_size=1,
    )
    assert background.bev_occupancy_grid_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


def test_background_name(config):
    """Test that background has correct name."""
    background = BEVOccupancyGridBg(config=config)
    assert background.name == "BEVOccupancyGridBg"


def test_background_config_access(config):
    """Test that background has access to config."""
    background = BEVOccupancyGridBg(config=config)
    assert background.config == config
    assert background.config.res == 0.05
    assert background.config.width == 50
    assert background.config.height == 60
