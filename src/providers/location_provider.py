# location_provider.py

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .gnss_provider import GnssProvider, UbxPvtRecord
from .uwb_provider import UwbProvider, UwbPosRecord
from .singleton import singleton


@dataclass
class LocationState:
    t_monotonic: float
    gnss: Optional[UbxPvtRecord]
    uwb_a: Optional[UwbPosRecord]
    uwb_b: Optional[UwbPosRecord]


@singleton
class LocationProvider:
    def __init__(
        self,
        *,
        gnss: GnssProvider,
        uwb0: UwbProvider,
        uwb1: UwbProvider,
    ) -> None:
        self._gnss = gnss
        self._uwb0 = uwb0
        self._uwb1 = uwb1

        self._state_lock = threading.Lock()
        now = time.monotonic()
        self._latest_state = LocationState(
            t_monotonic=now,
            gnss=None,
            uwb_a=None,
            uwb_b=None,
        )

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._gnss.start()
        self._uwb0.start()
        self._uwb1.start()

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="LocationProviderWorker")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return

        self._stop_evt.set()
        self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            self._thread = None

        self._uwb0.stop()
        self._uwb1.stop()
        self._gnss.stop()

    def get_state(self) -> LocationState:
        with self._state_lock:
            return self._latest_state

    @staticmethod
    def _gnss_tuple(rec: Optional[UbxPvtRecord]) -> Optional[Tuple[float, float, float]]:
        # (t, lat, lon)
        if rec is None or rec.lat is None or rec.lon is None:
            return None
        return (rec.t_monotonic, float(rec.lat), float(rec.lon))

    @staticmethod
    def _uwb_tuple(idx: int, rec: Optional[UwbPosRecord]) -> Optional[Tuple[int, float, float, float]]:
        # (idx, t, x, y)
        if rec is None or rec.x_m is None or rec.y_m is None:
            return None
        return (idx, rec.t_monotonic, float(rec.x_m), float(rec.y_m))

    def get(self) -> Dict[str, Any]:
        """
        Returns:
          - gnss: (t, lat, lon) or None
          - uwb:  [ (0, t, x, y) or None, (1, t, x, y) or None ]
        """
        st = self.get_state()
        g = self._gnss_tuple(st.gnss)
        u0 = self._uwb_tuple(0, st.uwb_a)
        u1 = self._uwb_tuple(1, st.uwb_b)
        return {"gnss": g, "uwb": [u0, u1]}

    def _run(self) -> None:
        next_t = time.monotonic()

        while not self._stop_evt.is_set():
            gnss = self._gnss.get_record()
            u0 = self._uwb0.get_record()
            u1 = self._uwb1.get_record()

            st = LocationState(
                t_monotonic=time.monotonic(),
                gnss=gnss,
                uwb_a=u0,
                uwb_b=u1,
            )
            with self._state_lock:
                self._latest_state = st

            next_t += 0.1
            dt = next_t - time.monotonic()
            if dt > 0:
                self._stop_evt.wait(dt)
            else:
                next_t = time.monotonic()
