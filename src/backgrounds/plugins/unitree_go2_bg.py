"""
Unitree Go2 Background - Go2 제어 Provider 초기화 및 시작.

Background에서 UnitreeGo2Provider를 초기화하고 start()하여,
Action Connector 등에서 제어 인터페이스를 사용할 수 있게 합니다.
"""

import logging
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_provider import UnitreeGo2Provider


class UnitreeGo2BgConfig(BackgroundConfig):
    """
    Configuration for Unitree Go2 Background.

    Parameters
    ----------
    unitree_ethernet : Optional[str]
        Unitree Go2 Ethernet channel (e.g. "eth0"). Passed to UnitreeGo2Provider for DDS.
    timeout : float
        RPC timeout in seconds for SportClient (default 10.0). Passed to UnitreeGo2Provider.
    """

    unitree_ethernet: Optional[str] = Field(
        default=None, description="Unitree Go2 Ethernet channel"
    )
    timeout: float = Field(
        default=1.0,
        description="RPC timeout in seconds for SportClient",
    )


class UnitreeGo2Bg(Background[UnitreeGo2BgConfig]):
    """
    Unitree Go2 Background.

    Initializes and starts UnitreeGo2Provider so that MoveConnector and other
    components can use the Go2 control interface (SDK, no ROS).
    """

    def __init__(self, config: UnitreeGo2BgConfig):
        super().__init__(config)

        channel = (self.config.unitree_ethernet or "").strip()
        timeout = self.config.timeout
        try:
            self.unitree_go2_provider = UnitreeGo2Provider(channel=channel, timeout=timeout)
            self.unitree_go2_provider.start()
            logging.info(
                "Unitree Go2 Provider initialized and started in background"
            )
        except Exception as e:
            logging.error(f"UnitreeGo2Bg: Failed to init UnitreeGo2Provider: {e}")
            raise

    def run(self) -> None:
        """
        Background process loop.

        Override if you need periodic work.
        """
        time.sleep(60)
