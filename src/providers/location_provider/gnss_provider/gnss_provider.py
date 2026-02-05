# gnss_provider.py

from __future__ import annotations

import threading
from typing import Optional, Dict, Any

from .ubx_thread import start_ubx_thread, UbxSharedState, UbxPvtRecord


class GnssProvider:
    def __init__(
        self,
        ser,
        measRate_ms: int = 100,
        write_lock: Optional[threading.RLock] = None
    ) -> None:
        self.ser = ser
        self.measRate_ms = measRate_ms
        self.write_lock = write_lock or threading.RLock()

        self._thread = None
        self._shared: Optional[UbxSharedState] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread, self._shared = start_ubx_thread(
            ser = self.ser,
            measRate_ms = self.measRate_ms,
            write_lock = self.write_lock
        )

    def stop(self) -> None:
        if self._thread is None:
            return

        self._thread.stop()
        self._thread = None
        self._shared = None

    def get(self) -> Optional[Dict[str, Any]]:
        if self._shared is None:
            return None

        rec: Optional[UbxPvtRecord] = self._shared.get_latest()
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
            "lon": rec.lon,
            "lat": rec.lat,
            "hAcc_m": rec.hAcc_m,
            "pDOP": rec.pDOP,
        }
