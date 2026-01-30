# ubx_thread.py

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from pyubx2 import UBXReader, UBXMessage, SET, UBX_PROTOCOL


@dataclass
class UbxPvtRecord:
    t_monotonic: float
    iTOW: Optional[int]
    fixType: Optional[int]
    diffSoln: Optional[int]
    carrSoln: Optional[int]
    numSV: Optional[int]
    lon: Optional[float]
    lat: Optional[float]
    hAcc_m: Optional[float]
    pDOP: Optional[float]


class UbxSharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_pvt: Optional[UbxPvtRecord] = None

    def set_pvt(self, rec: UbxPvtRecord) -> None:
        with self._lock:
            self._latest_pvt = rec

    def get_latest(self) -> Optional[UbxPvtRecord]:
        with self._lock:
            return self._latest_pvt


class UbxReaderThread(threading.Thread):
    def __init__(
        self,
        ser,
        shared: UbxSharedState,
        stop_evt: threading.Event,
        measRate_ms: int = 100,
        navRate: int = 1,
        timeRef: int = 0,
        write_lock: Optional[threading.RLock] = None,
        name: str = "UbxReader",
    ) -> None:
        super().__init__(daemon = True, name = name)
        self.ser = ser
        self.write_lock = write_lock or threading.RLock()
        self.shared = shared
        self.stop_evt = stop_evt

        self.measRate_ms = measRate_ms
        self.navRate = navRate
        self.timeRef = timeRef
    
    def stop(self) -> None:
        self.stop_evt.set()
        self.join(timeout = 2.0)

    def cfg_interface(self) -> None:
        with self.write_lock:
            self.ser.write(
                UBXMessage(
                    "CFG", "CFG-RATE", SET,
                    measRate=self.measRate_ms,
                    navRate=self.navRate,
                    timeRef=self.timeRef
                ).serialize()
            )
            time.sleep(0.1)

            self.ser.write(
                UBXMessage(
                    "CFG", "CFG-MSG", SET,
                    msgClass = 0x01,
                    msgID = 0x07,
                    rateUSB = 1
                ).serialize()
            )
            time.sleep(0.1)

    @staticmethod
    def parse_navpvt(parsed: UBXMessage) -> UbxPvtRecord:
        hAcc_mm = getattr(parsed, "hAcc", None) # milimeter
        hAcc_m = (hAcc_mm* 1e-3) if hAcc_mm is not None else None # meter

        return UbxPvtRecord(
            t_monotonic = time.monotonic(),
            iTOW = getattr(parsed, "iTOW", None),
            fixType = getattr(parsed, "fixType", None),
            diffSoln = getattr(parsed, "diffSoln", None),
            carrSoln = getattr(parsed, "carrSoln", None),
            numSV = getattr(parsed, "numSV", None),
            lon = getattr(parsed, "lon", None),
            lat = getattr(parsed, "lat", None),
            hAcc_m = hAcc_m,
            pDOP = getattr(parsed, "pDOP", None),
        )
    
    def run(self) -> None:
        self.cfg_interface()

        ubr = UBXReader(self.ser, protfilter=UBX_PROTOCOL)

        while not self.stop_evt.is_set():
            _, parsed = ubr.read()

            if isinstance(parsed, UBXMessage) and parsed.identity == "NAV-PVT":
                rec = self.parse_navpvt(parsed)
                self.shared.set_pvt(rec)

def start_ubx_thread(
    ser,
    measRate_ms: int = 100,
    write_lock: Optional[threading.RLock] = None
) -> Tuple[UbxReaderThread, UbxSharedState]:
    shared = UbxSharedState()
    stop_evt = threading.Event()
    th = UbxReaderThread(
        ser = ser,
        shared = shared,
        stop_evt = stop_evt,
        measRate_ms = measRate_ms,
        write_lock = write_lock
    )
    th.start()
    
    return th, shared