#!/usr/bin/env python3
"""
Tests for UnitreeGo2Provider and SDK verification.

- Pytest unit tests mock SportClient/ChannelFactoryInitialize so no robot is required.
- TestUnitreeGo2ProviderSDKAvailable.test_sdk_check_passes runs SDK verification (imports + SportClient APIs) when SDK is importable.
- Run from project root (src on path for unitree bridge):
    PYTHONPATH=src uv run pytest tests/providers/test_unitree_go2_provider.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_go2_provider import UnitreeGo2Provider


# ----- Pytest fixtures -----


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    UnitreeGo2Provider.reset()  # type: ignore
    yield
    UnitreeGo2Provider.reset()  # type: ignore


@pytest.fixture
def provider_params():
    """Default provider constructor params."""
    return {"channel": "", "timeout": 1.0}


@pytest.fixture
def mock_sport_client():
    """SportClient mock: all RPC methods return code 0, AutoRecoveryGet returns (0, None)."""
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


# ----- Initialization & singleton -----


class TestUnitreeGo2ProviderInitialization:
    """Test UnitreeGo2Provider initialization."""

    def test_default_initialization(self, provider_params):
        """Test initialization with default channel and timeout."""
        provider = UnitreeGo2Provider(**provider_params)

        assert provider._channel == ""
        assert provider._timeout == 10.0
        assert provider._sport_client is None
        assert not provider._running
        assert provider._data is None

    def test_custom_channel_and_timeout(self):
        """Test initialization with custom channel and timeout."""
        provider = UnitreeGo2Provider(channel="eth0", timeout=5.0)

        assert provider._channel == "eth0"
        assert provider._timeout == 5.0

    def test_singleton_pattern(self, provider_params):
        """Test that UnitreeGo2Provider is a singleton."""
        p1 = UnitreeGo2Provider(**provider_params)
        p2 = UnitreeGo2Provider(channel="other", timeout=3.0)

        assert p1 is p2


# ----- Lifecycle (start/stop) -----


class TestUnitreeGo2ProviderLifecycle:
    """Test start/stop lifecycle with mocked SDK."""

    def test_start_sets_sport_client_and_running(
        self, provider_params, mock_sport_client
    ):
        """Test start() initializes SportClient and sets _running."""
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()

        assert provider._running is True
        assert provider._sport_client is mock_sport_client
        assert provider._data == {"initialized": True}
        mock_sport_client.SetTimeout.assert_called_once_with(10.0)
        mock_sport_client.Init.assert_called_once()
        mock_sport_client.StopMove.assert_called_once()

    def test_start_with_channel_calls_channel_factory(
        self, mock_sport_client
    ):
        """Test start() with non-empty channel calls ChannelFactoryInitialize."""
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch(
                "providers.unitree_go2_provider.ChannelFactoryInitialize",
            ) as mock_channel_init,
        ):
            provider = UnitreeGo2Provider(channel="eth0", timeout=10.0)
            provider.start()

        mock_channel_init.assert_called_once_with(0, "eth0")

    def test_start_idempotent(self, provider_params, mock_sport_client):
        """Test calling start() twice does not re-init."""
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            provider.start()

        assert mock_sport_client.Init.call_count == 1

    def test_start_when_sport_client_unavailable(self, provider_params, caplog):
        """Test start() when SportClient import is None logs and does not set _running."""
        with patch("providers.unitree_go2_provider.SportClient", None):
            # Re-import would see None; we need to patch the already-imported module's
            # reference. The provider's start() checks SportClient is None.
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()

        assert not provider._running
        assert provider._sport_client is None
        assert "not available" in caplog.text

    def test_start_channel_init_failure(self, mock_sport_client, caplog):
        """Test start() when ChannelFactoryInitialize raises."""
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
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
        """Test start() when SportClient() or Init fails."""
        failing_client = MagicMock()
        failing_client.Init.side_effect = RuntimeError("DDS init failed")
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=failing_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()

        assert not provider._running
        assert provider._sport_client is None
        assert "Failed to init SportClient" in caplog.text

    def test_stop_clears_running_and_data(self, provider_params, mock_sport_client):
        """Test stop() sets _running False and _data None, calls StopMove."""
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            provider.stop()

        assert not provider._running
        assert provider._data is None
        assert mock_sport_client.StopMove.call_count >= 1

    def test_stop_handles_stop_move_exception(self, provider_params, mock_sport_client, caplog):
        """Test stop() logs but does not raise when StopMove raises (e.g. on second call)."""
        # First call (during start()) succeeds; second call (during stop()) raises
        mock_sport_client.StopMove.side_effect = [0, RuntimeError("connection lost")]
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            provider.stop()

        assert not provider._running
        assert "StopMove" in caplog.text


# ----- Data property -----


class TestUnitreeGo2ProviderData:
    """Test data property."""

    def test_data_none_before_start(self, provider_params):
        """Test data is None before start."""
        provider = UnitreeGo2Provider(**provider_params)
        assert provider.data is None

    def test_data_after_start(self, provider_params, mock_sport_client):
        """Test data is set after start."""
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
        assert provider.data == {"initialized": True}
        provider.stop()
        assert provider.data is None


# ----- Control methods (with mock_sport_client) -----


class TestUnitreeGo2ProviderMove:
    """Test move and stop_move."""

    def test_move_success(self, provider_params, mock_sport_client):
        """Test move() sends Move and returns True."""
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            ok = provider.move(0.2, 0.0, 0.1)
            provider.stop()

        assert ok is True
        mock_sport_client.Move.assert_called_once_with(0.2, 0.0, 0.1)

    def test_move_when_not_started(self, provider_params, caplog):
        """Test move() when _sport_client is None returns False."""
        provider = UnitreeGo2Provider(**provider_params)
        ok = provider.move(0.1, 0.0, 0.0)
        assert ok is False
        assert "not ready" in caplog.text

    def test_move_exception_returns_false(self, provider_params, mock_sport_client, caplog):
        """Test move() returns False and logs when Move raises."""
        mock_sport_client.Move.side_effect = RuntimeError("send failed")
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            ok = provider.move(0.0, 0.0, 0.0)
            provider.stop()
        assert ok is False
        assert "Move failed" in caplog.text

    def test_stop_move_success(self, provider_params, mock_sport_client):
        """Test stop_move() returns True when SDK returns 0."""
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            ok = provider.stop_move()
            provider.stop()
        assert ok is True
        mock_sport_client.StopMove.assert_called()

    def test_stop_move_when_not_started(self, provider_params):
        """Test stop_move() when _sport_client is None returns False."""
        provider = UnitreeGo2Provider(**provider_params)
        assert provider.stop_move() is False

    def test_stop_move_nonzero_code_returns_false(self, provider_params, mock_sport_client, caplog):
        """Test stop_move() returns False when SDK returns non-zero."""
        mock_sport_client.StopMove.return_value = -1
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            ok = provider.stop_move()
            provider.stop()
        assert ok is False
        assert "StopMove failed" in caplog.text


class TestUnitreeGo2ProviderPosture:
    """Test stand_up, stand_down, damp, sit, rise_sit, recovery_stand."""

    @pytest.fixture
    def started_provider(self, provider_params, mock_sport_client):
        """Provider with start() already called (mocked)."""
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            yield provider
            provider.stop()

    def test_stand_up_success(self, started_provider, mock_sport_client):
        """Test stand_up() returns True and calls StandUp."""
        assert started_provider.stand_up() is True
        mock_sport_client.StandUp.assert_called_once()

    def test_stand_down_success(self, started_provider, mock_sport_client):
        """Test stand_down() returns True and calls StandDown."""
        assert started_provider.stand_down() is True
        mock_sport_client.StandDown.assert_called_once()

    def test_damp_success(self, started_provider, mock_sport_client):
        """Test damp() returns True and calls Damp."""
        assert started_provider.damp() is True
        mock_sport_client.Damp.assert_called_once()

    def test_sit_success(self, started_provider, mock_sport_client):
        """Test sit() returns True and calls Sit."""
        assert started_provider.sit() is True
        mock_sport_client.Sit.assert_called_once()

    def test_rise_sit_success(self, started_provider, mock_sport_client):
        """Test rise_sit() returns True and calls RiseSit."""
        assert started_provider.rise_sit() is True
        mock_sport_client.RiseSit.assert_called_once()

    def test_recovery_stand_success(self, started_provider, mock_sport_client):
        """Test recovery_stand() returns True and calls RecoveryStand."""
        assert started_provider.recovery_stand() is True
        mock_sport_client.RecoveryStand.assert_called_once()

    def test_stand_up_when_not_started(self, provider_params):
        """Test stand_up() when _sport_client is None returns False."""
        provider = UnitreeGo2Provider(**provider_params)
        assert provider.stand_up() is False

    def test_stand_up_nonzero_code_returns_false(self, provider_params, mock_sport_client, caplog):
        """Test stand_up() returns False when SDK returns non-zero."""
        mock_sport_client.StandUp.return_value = -1
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            ok = provider.stand_up()
            provider.stop()
        assert ok is False
        assert "StandUp failed" in caplog.text


class TestUnitreeGo2ProviderHeartbeat:
    """Test heartbeat()."""

    def test_heartbeat_success(self, provider_params, mock_sport_client):
        """Test heartbeat() returns True when AutoRecoveryGet returns code 0."""
        mock_sport_client.AutoRecoveryGet.return_value = (0, None)
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            ok = provider.heartbeat()
            provider.stop()
        assert ok is True
        mock_sport_client.AutoRecoveryGet.assert_called_once()

    def test_heartbeat_when_not_started(self, provider_params):
        """Test heartbeat() when _sport_client is None returns False."""
        provider = UnitreeGo2Provider(**provider_params)
        assert provider.heartbeat() is False

    def test_heartbeat_nonzero_code_returns_false(self, provider_params, mock_sport_client):
        """Test heartbeat() returns False when AutoRecoveryGet returns non-zero code."""
        mock_sport_client.AutoRecoveryGet.return_value = (-1, None)
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            ok = provider.heartbeat()
            provider.stop()
        assert ok is False

    def test_heartbeat_exception_returns_false(self, provider_params, mock_sport_client):
        """Test heartbeat() returns False when AutoRecoveryGet raises."""
        mock_sport_client.AutoRecoveryGet.side_effect = RuntimeError("timeout")
        with (
            patch(
                "providers.unitree_go2_provider.SportClient",
                return_value=mock_sport_client,
            ),
            patch("providers.unitree_go2_provider.ChannelFactoryInitialize"),
        ):
            provider = UnitreeGo2Provider(**provider_params)
            provider.start()
            ok = provider.heartbeat()
            provider.stop()
        assert ok is False


# ----- SDK check as pytest test (optional, may be skipped if SDK not installed) -----


class TestUnitreeGo2ProviderSDKAvailable:
    """Tests that run only when SDK is importable (no robot required)."""

    def test_sdk_check_passes(self):
        """Verify Unitree SDK and CycloneDDS can be imported and SportClient has required APIs."""
        sport_client_apis_used_by_provider = (
            "SetTimeout",
            "Init",
            "StopMove",
            "Move",
            "StandUp",
            "StandDown",
            "Damp",
            "Sit",
            "RiseSit",
            "RecoveryStand",
            "AutoRecoveryGet",
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
            for attr in sport_client_apis_used_by_provider:
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
