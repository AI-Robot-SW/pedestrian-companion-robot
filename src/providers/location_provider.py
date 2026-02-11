# location_provider.py

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .gnss_provider import GnssProvider, UbxPvtRecord
from .uwb_provider import UwbProvider, UwbPosRecord
from .singleton import singleton

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocationState:
    t_monotonic: float
    gnss: Optional[UbxPvtRecord]
    uwb: Optional[UwbPosRecord]


@singleton
class LocationProvider:
    """
    Location Provider

    1개의 GNSS Provider와 2개의 UWB Provider에서 읽어온 최신 값을 하나로 합쳐 제공.

    Parameters
    ----------
    gnss : GnssProvider
        GNSS provider 인스턴스
    uwb0 : UwbProvider
        UWB provider 인스턴스
    """

    def __init__(
        self,
        *,
        gnss: GnssProvider,
        uwb: UwbProvider,
    ) -> None:
        self._gnss = gnss
        self._uwb = uwb

        self._state_lock = threading.Lock()
        now = time.monotonic()
        self._latest_state = LocationState(
            t_monotonic=now,
            gnss=None,
            uwb=None
        )

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running: bool = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._gnss.start()
        self._uwb.start()

        self._stop_evt.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="LocationProviderWorker",
        )
        self._thread.start()
        logger.info("LocationProvider started")

    def stop(self) -> None:
        if self._thread is None:
            return

        self.running = False
        self._stop_evt.set()

        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("LocationProvider worker thread did not stop within timeout")
        else:
            self._thread = None

        self._uwb.stop()
        self._gnss.stop()

        logger.info("LocationProvider stopped")

    def get_state(self) -> LocationState:
        with self._state_lock:
            return self._latest_state

    @staticmethod
    def _gnss_dict(rec: Optional[UbxPvtRecord]) -> Optional[Dict[str, Any]]:
        if rec is None:
            return None
        return {
            "t_monotonic": rec.t_monotonic,
            "hour": rec.hour,
            "minute": rec.minute,
            "second": rec.second,
            "validTime": rec.validTime,
            "fixType": rec.fixType,
            "diffSoln": rec.diffSoln,
            "carrSoln": rec.carrSoln,
            "numSV": rec.numSV,
            "lat": rec.lat,
            "lon": rec.lon,
            "hAcc_m": rec.hAcc_m,
            "pDOP": rec.pDOP,
        }

    @staticmethod
    def _uwb_dict(rec: Optional[UwbPosRecord]) -> Optional[Dict[str, Any]]:
        if rec is None:
            return None
        return {
            "t_monotonic": rec.t_monotonic,
            "x_m": rec.x_m,
            "y_m": rec.y_m,
            "z_m": rec.z_m,
            "quality_factor": rec.quality_factor,
        }

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        state = self.get_state()
        if state.gnss is None and state.uwb is None:
            return None

        return {
            "t_monotonic": state.t_monotonic,
            "gnss": self._gnss_dict(state.gnss),
            "uwb": self._uwb_dict(state.uwb),
        }

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                st = LocationState(
                    t_monotonic=time.monotonic(),
                    gnss = self._gnss.get_record(),
                    uwb = self._uwb.get_record()
                )
                with self._state_lock:
                    self._latest_state = st

            except Exception:
                logger.exception("Error in LocationProvider worker loop")
            
            self._stop_evt.wait(0.1)