from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.realsense_camera_bg import (
    RealSenseCameraBg,
    RealSenseCameraConfig,
)


@pytest.fixture
def config():
    return RealSenseCameraConfig(camera_index=0)


@pytest.fixture
def config_with_custom_index():
    return RealSenseCameraConfig(camera_index=1)


def test_config_initialization():
    """Test RealSenseCameraConfig initialization."""
    config = RealSenseCameraConfig()
    assert config.camera_index == 0

    config_custom = RealSenseCameraConfig(camera_index=1)
    assert config_custom.camera_index == 1


@patch("backgrounds.plugins.realsense_camera_bg.RealSenseCameraProvider")
def test_background_initialization(mock_provider_class, config):
    """Test RealSenseCameraBg initialization."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = RealSenseCameraBg(config=config)

    mock_provider_class.assert_called_once_with(camera_index=0)
    assert background.realsense_camera_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.realsense_camera_bg.RealSenseCameraProvider")
def test_background_initialization_custom_index(
    mock_provider_class, config_with_custom_index
):
    """Test RealSenseCameraBg initialization with custom camera_index."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = RealSenseCameraBg(config=config_with_custom_index)

    mock_provider_class.assert_called_once_with(camera_index=1)
    assert background.realsense_camera_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


def test_background_name(config):
    """Test that background has correct name."""
    background = RealSenseCameraBg(config=config)
    assert background.name == "RealSenseCameraBg"


def test_background_config_access(config):
    """Test that background has access to config."""
    background = RealSenseCameraBg(config=config)
    assert background.config == config
    assert background.config.camera_index == 0
