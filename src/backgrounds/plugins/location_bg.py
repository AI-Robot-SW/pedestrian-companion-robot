# src/backgrounds/plugins/location.py

from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.gnss_provider import GnssProvider
from providers.uwb_provider import UwbProvider
from providers.location_provider import LocationProvider

logger = logging.getLogger(__name__)


class LocationConfig(BackgroundConfig):
    gnss_port: Optional[str] = Field(default=None, description="GNSS serial port")
    gnss_baud: int = Field(default=115200, description="GNSS baudrate")

    uwb0_port: Optional[str] = Field(default=None, description="UWB0 serial port")
    uwb1_port: Optional[str] = Field(default=None, description="UWB1 serial port")
    uwb_baud: int = Field(default=115200, description="UWB baudrate")


class Location(Background[LocationConfig]):
    """
    Location Background.

    GNSS + UWB Provider들을 초기화하고, 이를 합치는 LocationProvider를 구동.
    """

    def __init__(self, config: LocationConfig):
        super().__init__(config)

        self.location_provider: Optional[LocationProvider] = None
        self._last_health_log_t = 0.0

        if self.config.gnss_port is None:
            logger.error("LocationConfig.gnss_port is not specified")
            return
        if self.config.uwb0_port is None or self.config.uwb1_port is None:
            logger.error("LocationConfig.uwb0_port / uwb1_port is not specified")
            return

        try:
            gnss = GnssProvider(serial_port=self.config.gnss_port, baudrate=self.config.gnss_baud)
            uwb0 = UwbProvider(serial_port=self.config.uwb0_port, baudrate=self.config.uwb_baud)
            uwb1 = UwbProvider(serial_port=self.config.uwb1_port, baudrate=self.config.uwb_baud)

            self.location_provider = LocationProvider(gnss=gnss, uwb0=uwb0, uwb1=uwb1)

            self.location_provider.start()

            logger.info(
                "Location background initialized providers: gnss=%s uwb0=%s uwb1=%s",
                self.config.gnss_port,
                self.config.uwb0_port,
                self.config.uwb1_port,
            )

        except Exception as e:
            logger.exception("Failed to initialize Location background: %s", e)
            self.location_provider = None

    def run(self) -> None:
        """
        Background tick.

        - busy-wait 방지만 수행.
        """
        if self.location_provider is None:
            time.sleep(1.0)
            return

        time.sleep(0.2)
