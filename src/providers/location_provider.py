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


@dataclass
class LocationState:
    """
    최신 위치 관련 레코드 스냅샷.

    Notes
    -----
    - 이 객체는 lock 보호 구간에서 교체되어 갱신.
    - 읽기 전용.
    """
    t_monotonic: float
    gnss: Optional[UbxPvtRecord]
    uwb_a: Optional[UwbPosRecord]
    uwb_b: Optional[UwbPosRecord]


@singleton
class LocationProvider:
    """
    Location Provider (기능/서비스 레벨 Provider).

    1개의 GNSS Provider와 2개의 UWB Provider에서 읽어온 최신 값을
    하나의 thread-safe 스냅샷으로 합쳐 제공.

    Classification
    -------------------
    - 기능/서비스 레벨 Provider:
      1개의 GNSS Provider과 2개의 UWB Provider를 조합하여
      통합 위치 상태를 제공.

    Parameters
    ----------
    gnss : GnssProvider
        GNSS provider 인스턴스
    uwb0 : UwbProvider
        UWB provider 인스턴스 (채널 A)
    uwb1 : UwbProvider
        UWB provider 인스턴스 (채널 B)
    """

    _PERIOD_S = 0.1  # 10 Hz update

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
        self.running: bool = False

    def start(self) -> None:
        """
        Provider 시작.

        - 하위 Provider(GNSS, UWB0, UWB1)를 먼저 start()
        - 이후 집계(aggregation) worker thread를 10 Hz로 실행
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._gnss.start()
        self._uwb0.start()
        self._uwb1.start()

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
        """
        Provider 정지.

        - worker thread 중지 신호를 보내고(join, timeout=2s)
        - 하위 Provider(UWB0, UWB1, GNSS) stop()
        """
        if self._thread is None:
            return

        self.running = False
        self._stop_evt.set()

        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("LocationProvider worker thread did not stop within timeout")
        else:
            self._thread = None

        self._uwb0.stop()
        self._uwb1.stop()
        self._gnss.stop()

        logger.info("LocationProvider stopped")

    def get_state(self) -> LocationState:
        """
        최신 스냅샷을 thread-safe하게 반환.

        Returns
        -------
        LocationState
            최신 스냅샷(읽기 전용으로 취급; 외부에서 mutate 금지)
        """
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
        (기존 호환) dict 형태로 통합 결과를 반환.

        Returns
        -------
        dict
            {
              "gnss": (t, lat, lon) or None,
              "uwb":  [ (0, t, x, y) or None, (1, t, x, y) or None ]
            }
        """
        st = self.get_state()
        g = self._gnss_tuple(st.gnss)
        u0 = self._uwb_tuple(0, st.uwb_a)
        u1 = self._uwb_tuple(1, st.uwb_b)
        return {"gnss": g, "uwb": [u0, u1]}

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        """
        Provider 표준 데이터 인터페이스(가이드 요구).

        Returns
        -------
        Optional[dict]
            현재 데이터(dict). 아직 데이터가 한 번도 들어온 적이 없으면 None.
        """
        st = self.get_state()
        if st.gnss is None and st.uwb_a is None and st.uwb_b is None:
            return None
        return self.get()

    def _run(self) -> None:
        """
        worker 루프(10 Hz).
        """
        next_t = time.monotonic()

        while not self._stop_evt.is_set():
            try:
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

            except Exception:
                logger.exception("Error in LocationProvider worker loop")

            next_t += self._PERIOD_S
            dt = next_t - time.monotonic()
            if dt > 0:
                self._stop_evt.wait(dt)
            else:
                next_t = time.monotonic()
