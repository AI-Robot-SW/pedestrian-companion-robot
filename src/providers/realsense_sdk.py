from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None
    _rs_import_error = e


@dataclass(frozen=True)
class FrameBundle:
    rgb: Dict[str, Any]
    depth: Dict[str, Any]
    camera_info: Dict[str, Any]
    timestamp: float


class RealSenseSDK:
    """
    RealSense SDK adapter
    - open/close
    - read() -> FrameBundle
    """
    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        align_depth_to_color: bool,
    ):
        if rs is None:
            raise ImportError(f"pyrealsense2 import failed: {_rs_import_error}")

        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.align_depth_to_color = bool(align_depth_to_color)

        self.pipeline: Optional["rs.pipeline"] = None
        self.config: Optional["rs.config"] = None
        self.profile: Optional["rs.pipeline_profile"] = None
        self.align: Optional["rs.align"] = None
        self.depth_scale: Optional[float] = None

    def open(self) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found.")
        if self.camera_index < 0 or self.camera_index >= len(devices):
            raise RuntimeError(f"camera_index out of range: {self.camera_index} / {len(devices)}")

        dev = devices[self.camera_index]
        serial = dev.get_info(rs.camera_info.serial_number)
        self.config.enable_device(serial)

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        self.align = rs.align(rs.stream.color) if self.align_depth_to_color else None

        # warm up
        for _ in range(10):
            self.pipeline.wait_for_frames()

    def close(self) -> None:
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        finally:
            self.pipeline = None
            self.config = None
            self.profile = None
            self.align = None

    def read(self, timestamp: float) -> FrameBundle:
        if self.pipeline is None:
            raise RuntimeError("RealSense SDK not started. Call open() first.")

        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Missing color or depth frame.")

        # RGB
        color_img = np.asanyarray(color_frame.get_data())
        h, w, _ = color_img.shape
        rgb = {
            "image": color_img,
            "encoding": "bgr8",
            "width": int(w),
            "height": int(h),
            "stride": int(w * 3),
        }

        # Depth
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        scale = float(self.depth_scale or 0.001)
        depth_m = depth_raw * scale
        dh, dw = depth_m.shape
        depth = {
            "image": depth_m,
            "encoding": "32FC1", #32-bit floating-point numbers, one channel
            "width": int(dw),
            "height": int(dh),
            "depth_scale": scale,
            "aligned_to": "color" if self.align is not None else "depth",
        }

        camera_info = self._read_camera_info()

        return FrameBundle(rgb=rgb, depth=depth, camera_info=camera_info, timestamp=timestamp)

    def _read_camera_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if self.profile is None:
            return info

        try:
            color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            ci = color_stream.get_intrinsics()
            info["color"] = {
                "width": int(ci.width),
                "height": int(ci.height),
                "fx": float(ci.fx),
                "fy": float(ci.fy),
                "cx": float(ci.ppx),
                "cy": float(ci.ppy),
                "model": str(ci.model),
                "coeffs": [float(x) for x in ci.coeffs],
            }
        except Exception:
            pass

        try:
            depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            di = depth_stream.get_intrinsics()
            info["depth"] = {
                "width": int(di.width),
                "height": int(di.height),
                "fx": float(di.fx),
                "fy": float(di.fy),
                "cx": float(di.ppx),
                "cy": float(di.ppy),
                "model": str(di.model),
                "coeffs": [float(x) for x in di.coeffs],
            }
        except Exception:
            pass

        return info
