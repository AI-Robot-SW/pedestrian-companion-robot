# uwb_provider.py

from __future__ import annotations

from typing import Optional, Dict, Any

from .uwb_thread import start_uwb_thread, UwbSharedState, UwbPosRecord


class UwbProvider:
    def __init__(self, ser) -> None:
        self.ser = ser

        self._thread = None
        self._shared: Optional[UwbSharedState] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread, self._shared = start_uwb_thread(ser=self.ser)

    def stop(self) -> None:
        if self._thread is None:
            return

        self._thread.stop()
        self._thread = None
        self._shared = None

    def get(self) -> Optional[Dict[str, Any]]:
        if self._shared is None:
            return None

        rec: Optional[UwbPosRecord] = self._shared.get_latest()
        if rec is None:
            return None

        return {
            "timestamp": rec.t_monotonic,
            "x_m": rec.x_m,
            "y_m": rec.y_m,
            "z_m": rec.z_m,
            "quality_factor": rec.quality_factor,
        }
