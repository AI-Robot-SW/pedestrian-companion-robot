"""
Tests for UnitreeGo2Bg (Unitree Go2 Background).

Mocks UnitreeGo2Provider so no robot or SDK connection is required.
"""

from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.unitree_go2_bg import UnitreeGo2Bg, UnitreeGo2BgConfig


@pytest.fixture
def config():
    """Default config: no ethernet channel, default timeout."""
    return UnitreeGo2BgConfig()


@pytest.fixture
def config_with_ethernet():
    """Config with ethernet channel and custom timeout."""
    return UnitreeGo2BgConfig(unitree_ethernet="eth0", timeout=10.0)


def test_config_initialization():
    """Test UnitreeGo2BgConfig initialization."""
    cfg = UnitreeGo2BgConfig()
    assert cfg.unitree_ethernet is None
    assert cfg.timeout == 1.0

    cfg_custom = UnitreeGo2BgConfig(unitree_ethernet="eth0", timeout=5.0)
    assert cfg_custom.unitree_ethernet == "eth0"
    assert cfg_custom.timeout == 5.0


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_initialization(mock_provider_class, config):
    """Test UnitreeGo2Bg initialization with default config."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = UnitreeGo2Bg(config=config)

    mock_provider_class.assert_called_once_with(channel="", timeout=1.0)
    assert background.unitree_go2_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_initialization_with_ethernet(mock_provider_class, config_with_ethernet):
    """Test UnitreeGo2Bg initialization with ethernet channel and custom timeout."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = UnitreeGo2Bg(config=config_with_ethernet)

    mock_provider_class.assert_called_once_with(channel="eth0", timeout=10.0)
    assert background.unitree_go2_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_initialization_strips_ethernet_whitespace(mock_provider_class):
    """Test that unitree_ethernet is stripped before passing to provider."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance
    config = UnitreeGo2BgConfig(unitree_ethernet="  eth0  ", timeout=2.0)

    UnitreeGo2Bg(config=config)

    mock_provider_class.assert_called_once_with(channel="eth0", timeout=2.0)


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_name(mock_provider_class, config):
    """Test that background has correct name."""
    mock_provider_class.return_value = MagicMock()
    background = UnitreeGo2Bg(config=config)
    assert background.name == "UnitreeGo2Bg"


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_config_access(mock_provider_class, config_with_ethernet):
    """Test that background has access to config."""
    mock_provider_class.return_value = MagicMock()
    background = UnitreeGo2Bg(config=config_with_ethernet)
    assert background.config == config_with_ethernet
    assert background.config.unitree_ethernet == "eth0"
    assert background.config.timeout == 10.0


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
@patch("backgrounds.plugins.unitree_go2_bg.time.sleep")
def test_run_sleeps(mock_sleep, mock_provider_class, config):
    """Test that run() calls time.sleep(60)."""
    mock_provider_class.return_value = MagicMock()
    background = UnitreeGo2Bg(config=config)

    background.run()

    mock_sleep.assert_called_once_with(60)


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_init_raises_on_provider_failure(mock_provider_class, config):
    """Test that __init__ raises when UnitreeGo2Provider fails."""
    mock_provider_class.side_effect = RuntimeError("DDS init failed")

    with pytest.raises(RuntimeError, match="DDS init failed"):
        UnitreeGo2Bg(config=config)
