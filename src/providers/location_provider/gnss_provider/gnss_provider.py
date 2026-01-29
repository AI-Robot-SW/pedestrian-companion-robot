# gnss_provider.py

from __future__ import annotations

from typing import Optional, Dict, Any

from .ubx_thread import start_ubx_thread, UbxSharedState, UbxPvtRecord


class GnssProvider:
    def __init__(
        self,
        ser,
        measRate_ms: int = 100,
    ) -> None:
        self.ser = ser
        self.measRate_ms = measRate_ms

        self._thread = None
        self._shared: Optional[UbxSharedState] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread, self._shared = start_ubx_thread(
            ser=self.ser,
            measRate_ms=self.measRate_ms,
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
            "timestamp": rec.t_monotonic,
            "fix_type": rec.fixType,
            "num_sv": rec.numSV,
            "lat": rec.lat,
            "lon": rec.lon,
            "h_acc_m": rec.hAcc_m,
            "p_dop": rec.pDOP,
        }
