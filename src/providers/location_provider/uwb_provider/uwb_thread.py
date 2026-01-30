# uwb_thread.py

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class UwbPosRecord:
    t_monotonic: float
    x_m: Optional[float]
    y_m: Optional[float]
    z_m: Optional[float]
    quality_factor: Optional[int]


class UwbSharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_pos: Optional[UwbPosRecord] = None

    def set_pos(self, rec: UwbPosRecord) -> None:
        with self._lock:
            self._latest_pos = rec

    def get_latest(self) -> Optional[UwbPosRecord]:
        with self._lock:
            return self._latest_pos


class UwbReaderThread(threading.Thread):
    def __init__(
        self,
        ser,
        shared: UwbSharedState,
        stop_evt: threading.Event,
        name: str = "UwbReader"
    ) -> None:
        super().__init__(daemon = True, name = name)
        self.ser = ser
        self._write_lock = threading.Lock()
        self.shared = shared
        self.stop_evt = stop_evt

    def stop(self) -> None:
        self.stop_evt.set()
        
        with self._write_lock:
            self.ser.write(b"\r")
            self.ser.flush()

        self.join(timeout=2.0)

    def read(self) -> bytes:
        n = getattr(self.ser, "in_waiting", 0) or 0
        if n > 0:
            return self.ser.read(n)
        else:
            return self.ser.read(1) # blocking 
        
    @staticmethod
    def extract_line(buf: bytearray) -> list[bytes]:
        out: list[bytes] = []
        start = 0

        while True:
            idx = buf.find(b"\r\n", start)
            if idx < 0:
                break

            line = bytes(buf[start:idx])
            if line:
                out.append(line)
                
            start = idx + 2

        if start:
            del buf[:start]

        return out

    def cfg_interface(self) -> None:
        with self._write_lock:
            self.ser.write(b"\r")
            time.sleep(0.1)
            self.ser.write(b"\r")
            time.sleep(1)
            self.ser.write(b"lep\r")
            time.sleep(0.1)

    @staticmethod
    def parse_lep(line: bytes) -> Optional[UwbPosRecord]:
        idx = line.find(b"POS,")

        if idx < 0:
            return None
        
        line = line[idx:]
        parts = line.split(b",")

        return UwbPosRecord(
            t_monotonic=time.monotonic(),
            x_m=float(parts[1]),
            y_m=float(parts[2]),
            z_m=float(parts[3]),
            quality_factor=int(float(parts[4])),
        )

    def run(self) -> None:
        self.cfg_interface()

        buf = bytearray()

        while not self.stop_evt.is_set():
            chunk = self.read()

            if chunk:
                buf.extend(chunk)
                for line in self.extract_line(buf):
                    rec = self.parse_lep(line)
                    if rec is not None:
                        self.shared.set_pos(rec)


def start_uwb_thread(
    ser
) -> Tuple[UwbReaderThread, UwbSharedState]:
    shared = UwbSharedState()
    stop_evt = threading.Event()
    th = UwbReaderThread(
        ser=ser,
        shared=shared,
        stop_evt=stop_evt
    )
    th.start()
    return th, shared
