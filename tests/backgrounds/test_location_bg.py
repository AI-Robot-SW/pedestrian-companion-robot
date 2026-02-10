# tests/backgrounds/test_location_bg.py

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.location_bg import Location, LocationConfig


@pytest.fixture
def config_default() -> LocationConfig:
    """Default config (ports are None; should fail validation and not init providers)."""
    return LocationConfig()


@pytest.fixture
def config_valid() -> LocationConfig:
    """Valid config for provider initialization (no real HW is used; providers are mocked)."""
    return LocationConfig(
        gnss_port="/dev/ttyACM0",
        gnss_baud=115200,
        uwb0_port="/dev/ttyACM1",
        uwb1_port="/dev/ttyACM2",
        uwb_baud=115200,
    )


def test_config_defaults(config_default: LocationConfig) -> None:
    assert config_default.gnss_port is None
    assert config_default.gnss_baud == 115200

    assert config_default.uwb0_port is None
    assert config_default.uwb1_port is None
    assert config_default.uwb_baud == 115200


def test_config_custom_values() -> None:
    cfg = LocationConfig(
        gnss_port="GNSS",
        gnss_baud=38400,
        uwb0_port="UWB0",
        uwb1_port="UWB1",
        uwb_baud=57600,
    )
    assert cfg.gnss_port == "GNSS"
    assert cfg.gnss_baud == 38400
    assert cfg.uwb0_port == "UWB0"
    assert cfg.uwb1_port == "UWB1"
    assert cfg.uwb_baud == 57600


@patch("backgrounds.plugins.location_bg.LocationProvider")
@patch("backgrounds.plugins.location_bg.UwbProvider")
@patch("backgrounds.plugins.location_bg.GnssProvider")
def test_background_initialization_calls_providers_and_starts(
    mock_gnss_cls,
    mock_uwb_cls,
    mock_locprov_cls,
    config_valid: LocationConfig,
) -> None:
    gnss_inst = MagicMock(name="gnss_inst")
    uwb0_inst = MagicMock(name="uwb0_inst")
    uwb1_inst = MagicMock(name="uwb1_inst")
    locprov_inst = MagicMock(name="location_provider_inst")

    mock_gnss_cls.return_value = gnss_inst
    mock_uwb_cls.side_effect = [uwb0_inst, uwb1_inst]
    mock_locprov_cls.return_value = locprov_inst

    bg = Location(config=config_valid)

    # Provider constructors called with args derived from config
    mock_gnss_cls.assert_called_once_with(serial_port=config_valid.gnss_port, baudrate=config_valid.gnss_baud)
    assert mock_uwb_cls.call_count == 2
    mock_uwb_cls.assert_any_call(serial_port=config_valid.uwb0_port, baudrate=config_valid.uwb_baud)
    mock_uwb_cls.assert_any_call(serial_port=config_valid.uwb1_port, baudrate=config_valid.uwb_baud)

    # Composite LocationProvider constructed with the provider instances
    mock_locprov_cls.assert_called_once_with(gnss=gnss_inst, uwb0=uwb0_inst, uwb1=uwb1_inst)

    # Background starts the composite provider once
    locprov_inst.start.assert_called_once()

    # Background stores the provider instance
    assert bg.location_provider is locprov_inst


@patch("backgrounds.plugins.location_bg.LocationProvider")
@patch("backgrounds.plugins.location_bg.UwbProvider")
@patch("backgrounds.plugins.location_bg.GnssProvider")
def test_name_and_config_access(
    mock_gnss_cls,
    mock_uwb_cls,
    mock_locprov_cls,
    config_valid: LocationConfig,
) -> None:
    mock_gnss_cls.return_value = MagicMock()
    mock_uwb_cls.side_effect = [MagicMock(), MagicMock()]
    mock_locprov_cls.return_value = MagicMock()

    bg = Location(config=config_valid)

    # Background base should preserve config object
    assert bg.config is config_valid

    # If runtime exposes name, it should match class name
    if hasattr(bg, "name"):
        assert bg.name == "Location"


@patch("backgrounds.plugins.location_bg.time.sleep")
def test_run_sleeps_when_provider_not_initialized(mock_sleep, config_default: LocationConfig) -> None:
    # No providers should be created; run should sleep(1.0) and return.
    bg = Location(config=config_default)
    assert bg.location_provider is None

    bg.run()
    mock_sleep.assert_called_once_with(1.0)


@patch("backgrounds.plugins.location_bg.time.sleep")
@patch("backgrounds.plugins.location_bg.LocationProvider")
@patch("backgrounds.plugins.location_bg.UwbProvider")
@patch("backgrounds.plugins.location_bg.GnssProvider")
def test_run_sleeps_when_provider_initialized(
    mock_gnss_cls,
    mock_uwb_cls,
    mock_locprov_cls,
    mock_sleep,
    config_valid: LocationConfig,
) -> None:
    # Init succeeds -> location_provider is set -> run should sleep(0.2)
    mock_gnss_cls.return_value = MagicMock()
    mock_uwb_cls.side_effect = [MagicMock(), MagicMock()]
    mock_locprov_cls.return_value = MagicMock()

    bg = Location(config=config_valid)
    assert bg.location_provider is not None

    bg.run()
    mock_sleep.assert_called_once_with(0.2)


@patch("backgrounds.plugins.location_bg.LocationProvider")
@patch("backgrounds.plugins.location_bg.UwbProvider")
@patch("backgrounds.plugins.location_bg.GnssProvider")
def test_init_failure_is_handled_and_provider_is_none(
    mock_gnss_cls,
    mock_uwb_cls,
    mock_locprov_cls,
    config_valid: LocationConfig,
) -> None:
    # Simulate failure during provider construction
    mock_gnss_cls.side_effect = RuntimeError("GNSS init failed")

    bg = Location(config=config_valid)

    # Background catches exception and sets location_provider to None
    assert bg.location_provider is None

    # If GNSS construction failed, these should not be called
    assert mock_uwb_cls.call_count == 0
    assert mock_locprov_cls.call_count == 0
