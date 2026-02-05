# test_ubx_thread.py

import time
import serial
import threading

from location_provider.gnss_provider.ubx_thread import start_ubx_thread


def main():
    ser = serial.Serial(
        port="/dev/ttyACM0",
        baudrate=115200,
        timeout=1.0
    )

    write_lock = threading.RLock()

    th, shared = start_ubx_thread(
        ser = ser,
        measRate_ms = 100,
        write_lock = write_lock
    )

    try:
        while True:
            rec = shared.get_latest()
            if rec:
                print(
                    f"[PVT] utc = {rec.hour}.{rec.minute}.{rec.second}, validTime={rec.validTime}", 
                    f"fix={rec.fixType}, "
                    f"sv={rec.numSV}, "
                    f"lat={rec.lat} lon={rec.lon}, "
                    f"hAcc={rec.hAcc_m:.2f}m, "
                    f"pDOP={rec.pDOP}"
                )
            else:
                print("[PVT] no data")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping GNSS thread...")
        th.stop()

    finally:
        ser.close()


if __name__ == "__main__":
    main()
