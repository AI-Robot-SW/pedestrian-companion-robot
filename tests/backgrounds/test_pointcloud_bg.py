from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.pointcloud_bg import PointCloudBg, PointCloudConfig


@pytest.fixture
def config():
    return PointCloudConfig(range_max=3.0, stride=2, sync_slop=0.05)


@pytest.fixture
def config_default():
    return PointCloudConfig()


def test_config_initialization():
    """Test PointCloudConfig initialization."""
    config = PointCloudConfig()
    assert config.range_max == 0.0
    assert config.stride == 1
    assert config.sync_slop == 0.05

    config_custom = PointCloudConfig(range_max=5.0, stride=3, sync_slop=0.1)
    assert config_custom.range_max == 5.0
    assert config_custom.stride == 3
    assert config_custom.sync_slop == 0.1


@patch("backgrounds.plugins.pointcloud_bg.PointCloudProvider")
def test_background_initialization(mock_provider_class, config):
    """Test PointCloudBg initialization."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = PointCloudBg(config=config)

    mock_provider_class.assert_called_once_with(
        range_max=3.0, stride=2, sync_slop=0.05
    )
    assert background.pointcloud_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.pointcloud_bg.PointCloudProvider")
def test_background_initialization_default(mock_provider_class, config_default):
    """Test PointCloudBg initialization with default parameters."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = PointCloudBg(config=config_default)

    mock_provider_class.assert_called_once_with(
        range_max=0.0, stride=1, sync_slop=0.05
    )
    assert background.pointcloud_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


def test_background_name(config):
    """Test that background has correct name."""
    background = PointCloudBg(config=config)
    assert background.name == "PointCloudBg"


def test_background_config_access(config):
    """Test that background has access to config."""
    background = PointCloudBg(config=config)
    assert background.config == config
    assert background.config.range_max == 3.0
    assert background.config.stride == 2
    assert background.config.sync_slop == 0.05
