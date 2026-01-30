# test_uwb_thread.py

import time
import serial

from location_provider.uwb_provider.uwb_thread import start_uwb_thread


def main():
    ser = serial.Serial(
        port="/dev/ttyACM1",
        baudrate=115200,
        timeout=1.0
    )

    th, shared = start_uwb_thread(ser)

    try:
        while True:
            rec = shared.get_latest()
            if rec:
                print(f"[UWB] x={rec.x_m:.3f}, "
                      f"y={rec.y_m:.3f}, "
                      f"z={rec.z_m:.3f}, "
                      f"qf={rec.y_m:.3f}, "
                      f"qf={rec.y_m:.3f}, "
                      f"t={rec.t_monotonic:.3f}"
                )
            else:
                print("[UWB] no data")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping UWB thread...")
        th.stop()

    finally:
        ser.close()

if __name__ == "__main__":
    main()
