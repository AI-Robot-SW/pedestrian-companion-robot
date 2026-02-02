import logging
import threading
import time
from typing import Optional

from .singleton import singleton
from .realsense_sdk import RealSenseSDK, FrameBundle


@singleton
class RealSenseCameraProvider:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_depth_to_color: bool = True,
    ):
        logging.info(f"RealSenseCameraProvider booting at camera_index: {camera_index}")

        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.align_depth_to_color = bool(align_depth_to_color)

        # tests expect these
        self._rgb_data: Optional[dict] = None
        self._depth_data: Optional[dict] = None
        self._camera_info: Optional[dict] = None
        self._data: Optional[dict] = None

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # SDK adapter (separated)
        self._sdk: Optional[RealSenseSDK] = None

        logging.info("RealSenseCameraProvider initialized")

    def start(self):
        if self._thread and self._thread.is_alive():
            logging.warning("RealSenseCameraProvider already running")
            return

        # keep method name for tests (they patch _start_sdk)
        self._start_sdk()

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("RealSenseCameraProvider started")

    def stop(self):
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None

        # keep method name for symmetry / tests
        self._stop_sdk()

        logging.info("RealSenseCameraProvider stopped")

    def _run(self):
        period = 1.0 / max(1, self.fps)

        while self.running:
            t0 = time.time()
            try:
                self._update_data()
            except Exception as e:
                logging.error(f"Error in RealSenseCameraProvider loop: {e}")
                with self._lock:
                    self._data = None
                    self._rgb_data = None
                    self._depth_data = None
                    self._camera_info = None

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

# SDK lifecycle (kept for tests)
    def _start_sdk(self):
        if self._sdk is not None:
            return

        self._sdk = RealSenseSDK(
            camera_index=self.camera_index,
            width=self.width,
            height=self.height,
            fps=self.fps,
            align_depth_to_color=self.align_depth_to_color,
        )
        self._sdk.open()

    def _stop_sdk(self):
        if self._sdk is None:
            return
        try:
            self._sdk.close()
        finally:
            self._sdk = None

# Data update
    def _update_data(self):
        if self._sdk is None:
            raise RuntimeError("SDK not started. Call start() first.")

        ts = time.time()
        bundle: FrameBundle = self._sdk.read(timestamp=ts)

        # keep these internal fields for tests
        self._rgb_data = bundle.rgb
        self._depth_data = bundle.depth
        self._camera_info = bundle.camera_info

        with self._lock:
            self._data = {
                "rgb": self._rgb_data,
                "depth": self._depth_data,
                "camera_info": self._camera_info,
                "timestamp": bundle.timestamp,
            }

    @property
    def data(self) -> Optional[dict]:
        with self._lock:
            return self._data
