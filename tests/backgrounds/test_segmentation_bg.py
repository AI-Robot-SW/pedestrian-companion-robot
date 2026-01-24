from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.segmentation_bg import SegmentationBg, SegmentationConfig


@pytest.fixture
def config():
    return SegmentationConfig(engine_path="/path/to/engine.engine")


@pytest.fixture
def config_default():
    return SegmentationConfig()


def test_config_initialization():
    """Test SegmentationConfig initialization."""
    config = SegmentationConfig()
    assert config.engine_path == ""

    config_custom = SegmentationConfig(engine_path="/custom/path.engine")
    assert config_custom.engine_path == "/custom/path.engine"


@patch("backgrounds.plugins.segmentation_bg.SegmentationProvider")
def test_background_initialization(mock_provider_class, config):
    """Test SegmentationBg initialization."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = SegmentationBg(config=config)

    mock_provider_class.assert_called_once_with(engine_path="/path/to/engine.engine")
    assert background.segmentation_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.segmentation_bg.SegmentationProvider")
def test_background_initialization_default(mock_provider_class, config_default):
    """Test SegmentationBg initialization with default engine_path."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = SegmentationBg(config=config_default)

    mock_provider_class.assert_called_once_with(engine_path="")
    assert background.segmentation_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


def test_background_name(config):
    """Test that background has correct name."""
    background = SegmentationBg(config=config)
    assert background.name == "SegmentationBg"


def test_background_config_access(config):
    """Test that background has access to config."""
    background = SegmentationBg(config=config)
    assert background.config == config
    assert background.config.engine_path == "/path/to/engine.engine"
