#!/usr/bin/env python3
"""
Tests for UnitreeGo2Provider.

Follows .cursor/skills/provider-testing/SKILL.md:
- Mock SportClient and ChannelFactoryInitialize (no robot required).
- Fixtures: reset_singleton (autouse), provider_params, mock_sport_client.
- Test: initialization, singleton, lifecycle, data, control/API methods, error paths, SDK check.

Run: uv run pytest tests/providers/test_unitree_go2_provider.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_provider import UnitreeGo2Provider


# ----- Fixtures -----


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    UnitreeGo2Provider.reset()  # type: ignore
    yield
    UnitreeGo2Provider.reset()  # type: ignore


@pytest.fixture
def provider_params():
    """Default provider constructor params (matches implementation defaults)."""
    return {"channel": "", "timeout": 10.0}


@pytest.fixture
def mock_sport_client():
    """SportClient mock: RPC methods return 0, AutoRecoveryGet returns (0, None)."""
    client = MagicMock()
    client.SetTimeout.return_value = None
    client.Init.return_value = None
    client.StopMove.return_value = 0
    client.Move.return_value = None
    client.StandUp.return_value = 0
    client.StandDown.return_value = 0
    client.Damp.return_value = 0
    client.Sit.return_value = 0
    client.RiseSit.return_value = 0
    client.RecoveryStand.return_value = 0
    client.AutoRecoveryGet.return_value = (0, None)
    return client


def _start_provider(provider_params, mock_sport_client, channel=None, timeout=None):
    """Helper: patch SportClient and ChannelFactoryInitialize, create provider and start()."""
    kwargs = dict(provider_params)
    if channel is not None:
        kwargs["channel"] = channel
    if timeout is not None:
        kwargs["timeout"] = timeout
    with (
        patch("providers.unitree_go2_provider.SportClient", return_value=mock_sport_client),
        patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
    ):
        provider = UnitreeGo2Provider(**kwargs)
        provider.start()
        return provider


# ----- Initialization -----


class TestUnitreeGo2ProviderInitialization:
    """Constructor stores params; internal state correct."""

    def test_default_initialization(self, provider_params):
        provider = UnitreeGo2Provider(**provider_params)
        assert provider._channel == ""
        assert provider._timeout == 10.0
        assert provider._sport_client is None
        assert not provider._running
        assert provider._data is None

    def test_custom_channel_and_timeout(self):
        provider = UnitreeGo2Provider(channel="eth0", timeout=5.0)
        assert provider._channel == "eth0"
        assert provider._timeout == 5.0

    def test_singleton_pattern(self, provider_params):
        p1 = UnitreeGo2Provider(**provider_params)
        p2 = UnitreeGo2Provider(channel="other", timeout=3.0)
        assert p1 is p2


# ----- Lifecycle -----


class TestUnitreeGo2ProviderLifecycle:
    """start() sets state and init client; stop() clears state; idempotent start()."""

    def test_start_sets_sport_client_and_running(self, provider_params, mock_sport_client):
        provider = _start_provider(provider_params, mock_sport_client)
        assert provider._running is True
        assert provider._sport_client is mock_sport_client
        assert provider._data == {"initialized": True}
        mock_sport_client.SetTimeout.assert_called_once_with(10.0)
        mock_sport_client.Init.assert_called_once()
        mock_sport_client.StopMove.assert_called_once()
        provider.stop()

    def test_start_with_channel_calls_channel_factory(self, mock_sport_client):
        with (
            patch("providers.unitree_go2_provider.SportClient", return_value=mock_sport_client),
            patch(
                "providers.unitree_go2_provider.ChannelFactoryInitialize",
            ) as mock_channel_init,
        ):
            provider = UnitreeGo2Provider(channel="eth0", timeout=10.0)
            provider.start()
        mock_channel_init.assert_called_once_with(0, "eth0")
        provider.stop()

    def test_start_idempotent(self, provider_params, mock_sport_client):
        provider = _start_provider(provider_params, mock_sport_client)
        provider.start()
        assert mock_sport_client.Init.call_count == 1
        provider.stop()

    def test_start_when_sport_client_unavailable(self, provider_params, caplog):
        with patch("providers.unitree_go2_provider.SportClient", None):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
        assert not provider._running
        assert provider._sport_client is None
        assert "not available" in caplog.text

    def test_start_channel_init_failure(self, mock_sport_client, caplog):
        with (
            patch("providers.unitree_go2_provider.SportClient", return_value=mock_sport_client),
            patch(
                "providers.unitree_go2_provider.ChannelFactoryInitialize",
                side_effect=OSError("no eth0"),
            ),
        ):
            provider = UnitreeGo2Provider(channel="eth0", timeout=10.0)
            provider.start()
        assert not provider._running
        assert provider._sport_client is None
        assert "ChannelFactoryInitialize" in caplog.text

    def test_start_sport_client_init_failure(self, provider_params, caplog):
        failing_client = MagicMock()
        failing_client.Init.side_effect = RuntimeError("DDS init failed")
        with (
            patch("providers.unitree_go2_provider.SportClient", return_value=failing_client),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
        assert not provider._running
        assert provider._sport_client is None
        assert "Failed to init SportClient" in caplog.text

    def test_stop_clears_running_and_data(self, provider_params, mock_sport_client):
        provider = _start_provider(provider_params, mock_sport_client)
        provider.stop()
        assert not provider._running
        assert provider._data is None
        assert mock_sport_client.StopMove.call_count >= 1

    def test_stop_handles_stop_move_exception(self, provider_params, mock_sport_client, caplog):
        mock_sport_client.StopMove.side_effect = [0, RuntimeError("connection lost")]
        provider = _start_provider(provider_params, mock_sport_client)
        provider.stop()
        assert not provider._running
        assert "StopMove" in caplog.text


# ----- Data -----


class TestUnitreeGo2ProviderData:
    """data property before/after start and after stop."""

    def test_data_none_before_start(self, provider_params):
        provider = UnitreeGo2Provider(**provider_params)
        assert provider.data is None

    def test_data_after_start(self, provider_params, mock_sport_client):
        provider = _start_provider(provider_params, mock_sport_client)
        assert provider.data == {"initialized": True}
        provider.stop()
        assert provider.data is None


# ----- Control/API: move, stop_move -----


class TestUnitreeGo2ProviderMove:
    """move() and stop_move() call client with correct args and return True/False."""

    def test_move_success(self, provider_params, mock_sport_client):
        provider = _start_provider(provider_params, mock_sport_client)
        ok = provider.move(0.2, 0.0, 0.1)
        provider.stop()
        assert ok is True
        mock_sport_client.Move.assert_called_once_with(0.2, 0.0, 0.1)

    def test_move_when_not_started(self, provider_params, caplog):
        provider = UnitreeGo2Provider(**provider_params)
        ok = provider.move(0.1, 0.0, 0.0)
        assert ok is False
        assert "not ready" in caplog.text

    def test_move_exception_returns_false(self, provider_params, mock_sport_client, caplog):
        mock_sport_client.Move.side_effect = RuntimeError("send failed")
        provider = _start_provider(provider_params, mock_sport_client)
        ok = provider.move(0.0, 0.0, 0.0)
        provider.stop()
        assert ok is False
        assert "Move failed" in caplog.text

    def test_stop_move_success(self, provider_params, mock_sport_client):
        provider = _start_provider(provider_params, mock_sport_client)
        ok = provider.stop_move()
        provider.stop()
        assert ok is True
        mock_sport_client.StopMove.assert_called()

    def test_stop_move_when_not_started(self, provider_params):
        provider = UnitreeGo2Provider(**provider_params)
        assert provider.stop_move() is False

    def test_stop_move_nonzero_code_returns_false(self, provider_params, mock_sport_client, caplog):
        mock_sport_client.StopMove.return_value = -1
        provider = _start_provider(provider_params, mock_sport_client)
        ok = provider.stop_move()
        provider.stop()
        assert ok is False
        assert "StopMove failed" in caplog.text


# ----- Control/API: posture (stand_up, stand_down, damp, sit, rise_sit, recovery_stand) -----


class TestUnitreeGo2ProviderPosture:
    """Posture methods call client and return True/False."""

    @pytest.fixture
    def started_provider(self, provider_params, mock_sport_client):
        with (
            patch("providers.unitree_go2_provider.SportClient", return_value=mock_sport_client),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            yield provider
            provider.stop()

    def test_stand_up_success(self, started_provider, mock_sport_client):
        assert started_provider.stand_up() is True
        mock_sport_client.StandUp.assert_called_once()

    def test_stand_down_success(self, started_provider, mock_sport_client):
        assert started_provider.stand_down() is True
        mock_sport_client.StandDown.assert_called_once()

    def test_damp_success(self, started_provider, mock_sport_client):
        assert started_provider.damp() is True
        mock_sport_client.Damp.assert_called_once()

    def test_sit_success(self, started_provider, mock_sport_client):
        assert started_provider.sit() is True
        mock_sport_client.Sit.assert_called_once()

    def test_rise_sit_success(self, started_provider, mock_sport_client):
        assert started_provider.rise_sit() is True
        mock_sport_client.RiseSit.assert_called_once()

    def test_recovery_stand_success(self, started_provider, mock_sport_client):
        assert started_provider.recovery_stand() is True
        mock_sport_client.RecoveryStand.assert_called_once()

    def test_stand_up_when_not_started(self, provider_params):
        provider = UnitreeGo2Provider(**provider_params)
        assert provider.stand_up() is False

    def test_stand_up_nonzero_code_returns_false(self, provider_params, mock_sport_client, caplog):
        mock_sport_client.StandUp.return_value = -1
        provider = _start_provider(provider_params, mock_sport_client)
        ok = provider.stand_up()
        provider.stop()
        assert ok is False
        assert "StandUp failed" in caplog.text


# ----- Control/API: heartbeat -----


class TestUnitreeGo2ProviderHeartbeat:
    """heartbeat() calls AutoRecoveryGet and returns True when code==0."""

    def test_heartbeat_success(self, provider_params, mock_sport_client):
        provider = _start_provider(provider_params, mock_sport_client)
        ok = provider.heartbeat()
        provider.stop()
        assert ok is True
        mock_sport_client.AutoRecoveryGet.assert_called_once()

    def test_heartbeat_when_not_started(self, provider_params):
        provider = UnitreeGo2Provider(**provider_params)
        assert provider.heartbeat() is False

    def test_heartbeat_nonzero_code_returns_false(self, provider_params, mock_sport_client):
        mock_sport_client.AutoRecoveryGet.return_value = (-1, None)
        provider = _start_provider(provider_params, mock_sport_client)
        ok = provider.heartbeat()
        provider.stop()
        assert ok is False

    def test_heartbeat_exception_returns_false(self, provider_params, mock_sport_client):
        mock_sport_client.AutoRecoveryGet.side_effect = RuntimeError("timeout")
        provider = _start_provider(provider_params, mock_sport_client)
        ok = provider.heartbeat()
        provider.stop()
        assert ok is False


# ----- SDK/import check (runs when SDK importable, no robot) -----


class TestUnitreeGo2ProviderSDKAvailable:
    """Verify SDK and SportClient APIs when SDK is importable."""

    def test_sdk_check_passes(self):
        """cyclonedds, ChannelFactoryInitialize, SportClient + required APIs."""
        apis = (
            "SetTimeout", "Init", "StopMove", "Move",
            "StandUp", "StandDown", "Damp", "Sit", "RiseSit",
            "RecoveryStand", "AutoRecoveryGet",
        )
        all_ok = True

        try:
            import cyclonedds  # noqa: F401
            print("cyclonedds OK")
        except ImportError as e:
            print(f"cyclonedds FAIL: {e}")
            all_ok = False

        try:
            from unitree.unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: F401
            print("ChannelFactoryInitialize OK")
        except ImportError as e:
            print(f"ChannelFactoryInitialize FAIL: {e}")
            all_ok = False

        try:
            from unitree.unitree_sdk2py.go2.sport.sport_client import SportClient
            for attr in apis:
                if not hasattr(SportClient, attr) or not callable(getattr(SportClient, attr)):
                    print(f"SportClient.{attr} MISSING or not callable")
                    all_ok = False
            else:
                print("SportClient + required APIs OK")
        except ImportError as e:
            print(f"SportClient FAIL: {e}")
            all_ok = False

        if all_ok:
            print("SDK check: cyclonedds, ChannelFactoryInitialize, SportClient APIs — all OK")
        assert all_ok is True
