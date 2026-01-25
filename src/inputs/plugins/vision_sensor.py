"""
Vision Sensor - Input Orchestrator Vision 모듈

이 모듈은 카메라 입력을 통해 시각 데이터를 수집하고,
VLM(Vision Language Model) 결과를 Fuser에 전달합니다.

Architecture:
    RealSenseCameraBg -> RealSenseCameraProvider -> VisionSensor -> InputOrchestrator -> Fuser
    SegmentationBg -> SegmentationProvider ─┘
    BEVOccupancyGridBg -> BEVOccupancyGridProvider ─┘
    PointCloudBg -> PointCloudProvider ─┘

Dependencies:
    - RealSenseCameraProvider: RGB-D 카메라 스트림 관리
    - SegmentationProvider: 이미지 세그멘테이션
    - BEVOccupancyGridProvider: Bird's Eye View 점유 그리드
    - PointCloudProvider: 3D 포인트 클라우드
    - VLMProvider: Vision Language Model 추론
"""

import asyncio
import logging
import time
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput

# TODO: Provider import 추가 예정
# from providers.realsense_camera_provider import RealSenseCameraProvider
# from providers.segmentation_provider import SegmentationProvider
# from providers.bev_occupancy_grid_provider import BEVOccupancyGridProvider
# from providers.point_cloud_provider import PointCloudProvider
# from providers.vlm_provider import VLMProvider


class VisionSensorConfig(SensorConfig):
    """
    Vision Sensor 설정.

    Parameters
    ----------
    camera_device_id : Optional[int]
        카메라 디바이스 ID
    camera_serial : Optional[str]
        RealSense 카메라 시리얼 번호
    width : int
        이미지 너비 (pixels)
    height : int
        이미지 높이 (pixels)
    fps : int
        프레임 레이트
    enable_depth : bool
        깊이 스트림 활성화 여부
    enable_segmentation : bool
        세그멘테이션 활성화 여부
    enable_bev : bool
        BEV 점유 그리드 활성화 여부
    enable_pointcloud : bool
        포인트 클라우드 활성화 여부
    vlm_model : str
        VLM 모델 이름
    vlm_prompt : Optional[str]
        VLM 프롬프트 템플릿
    inference_interval_ms : int
        VLM 추론 간격 (ms)
    """

    camera_device_id: Optional[int] = Field(default=None, description="카메라 디바이스 ID")
    camera_serial: Optional[str] = Field(
        default=None, description="RealSense 카메라 시리얼"
    )
    width: int = Field(default=640, description="이미지 너비")
    height: int = Field(default=480, description="이미지 높이")
    fps: int = Field(default=30, description="프레임 레이트")
    enable_depth: bool = Field(default=True, description="깊이 스트림 활성화")
    enable_segmentation: bool = Field(default=False, description="세그멘테이션 활성화")
    enable_bev: bool = Field(default=False, description="BEV 점유 그리드 활성화")
    enable_pointcloud: bool = Field(default=False, description="포인트 클라우드 활성화")
    vlm_model: str = Field(default="gemini-pro-vision", description="VLM 모델")
    vlm_prompt: Optional[str] = Field(default=None, description="VLM 프롬프트 템플릿")
    inference_interval_ms: int = Field(default=1000, description="VLM 추론 간격 (ms)")


class VisionData:
    """
    Vision 데이터 컨테이너.

    Attributes
    ----------
    timestamp : float
        데이터 타임스탬프
    rgb_frame : Optional[np.ndarray]
        RGB 이미지 프레임
    depth_frame : Optional[np.ndarray]
        깊이 이미지 프레임
    segmentation_mask : Optional[np.ndarray]
        세그멘테이션 마스크
    bev_grid : Optional[np.ndarray]
        BEV 점유 그리드
    pointcloud : Optional[np.ndarray]
        3D 포인트 클라우드
    vlm_description : Optional[str]
        VLM 장면 설명
    detected_objects : List[Dict[str, Any]]
        감지된 객체 목록
    """

    def __init__(
        self,
        timestamp: float,
        rgb_frame: Optional[np.ndarray] = None,
        depth_frame: Optional[np.ndarray] = None,
        segmentation_mask: Optional[np.ndarray] = None,
        bev_grid: Optional[np.ndarray] = None,
        pointcloud: Optional[np.ndarray] = None,
        vlm_description: Optional[str] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
    ):
        self.timestamp = timestamp
        self.rgb_frame = rgb_frame
        self.depth_frame = depth_frame
        self.segmentation_mask = segmentation_mask
        self.bev_grid = bev_grid
        self.pointcloud = pointcloud
        self.vlm_description = vlm_description
        self.detected_objects = detected_objects or []


class VisionSensor(FuserInput[VisionSensorConfig, Optional[VisionData]]):
    """
    Vision Sensor - 시각 입력 처리 센서.

    카메라로부터 이미지 데이터를 수집하고, VLM을 통해 장면 설명을 생성하여
    InputOrchestrator에 전달합니다.

    Attributes
    ----------
    descriptor_for_LLM : str
        LLM에 전달될 입력 소스 설명자
    message_buffer : Queue[VisionData]
        Vision 데이터 버퍼
    latest_description : Optional[str]
        최신 VLM 장면 설명

    Methods
    -------
    _poll() -> Optional[VisionData]
        버퍼에서 새 Vision 데이터 폴링
    _raw_to_text(raw_input) -> Optional[Message]
        Vision 데이터를 텍스트 메시지로 변환
    formatted_latest_buffer() -> Optional[str]
        최신 버퍼 내용을 포맷팅하여 반환
    """

    def __init__(self, config: VisionSensorConfig):
        """
        Vision Sensor 초기화.

        Parameters
        ----------
        config : VisionSensorConfig
            센서 설정
        """
        super().__init__(config)

        # LLM 입력 설명자
        self.descriptor_for_LLM = "Vision"

        # 데이터 버퍼
        self.message_buffer: Queue[VisionData] = Queue()
        self.latest_description: Optional[str] = None
        self.latest_objects: List[Dict[str, Any]] = []

        # TODO: Provider 초기화
        # self.camera_provider = RealSenseCameraProvider(...)
        # self.segmentation_provider = SegmentationProvider(...)
        # self.bev_provider = BEVOccupancyGridProvider(...)
        # self.pointcloud_provider = PointCloudProvider(...)
        # self.vlm_provider = VLMProvider(...)

        logging.info(
            f"VisionSensor initialized with resolution={config.width}x{config.height}, "
            f"fps={config.fps}, vlm_model={config.vlm_model}"
        )

    def _handle_vlm_result(self, description: str, objects: List[Dict[str, Any]]) -> None:
        """
        VLM 결과 콜백 핸들러.

        Parameters
        ----------
        description : str
            VLM 장면 설명
        objects : List[Dict[str, Any]]
            감지된 객체 목록
        """
        vision_data = VisionData(
            timestamp=time.time(),
            vlm_description=description,
            detected_objects=objects,
        )
        self.message_buffer.put(vision_data)
        logging.debug(f"VLM result received: {description[:100]}...")

    async def _poll(self) -> Optional[VisionData]:
        """
        메시지 버퍼에서 새 Vision 데이터를 폴링.

        Returns
        -------
        Optional[VisionData]
            버퍼에 데이터가 있으면 반환, 없으면 None
        """
        await asyncio.sleep(0.1)
        try:
            vision_data = self.message_buffer.get_nowait()
            return vision_data
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[VisionData]) -> Optional[Message]:
        """
        Vision 데이터를 Message 객체로 변환.

        Parameters
        ----------
        raw_input : Optional[VisionData]
            Vision 데이터

        Returns
        -------
        Optional[Message]
            타임스탬프가 포함된 Message 객체
        """
        if raw_input is None:
            return None

        # VLM 설명을 메시지로 변환
        message_parts = []

        if raw_input.vlm_description:
            message_parts.append(f"Scene: {raw_input.vlm_description}")

        if raw_input.detected_objects:
            objects_str = ", ".join(
                [obj.get("label", "unknown") for obj in raw_input.detected_objects]
            )
            message_parts.append(f"Objects: {objects_str}")

        message = " | ".join(message_parts) if message_parts else "No visual data"

        return Message(timestamp=raw_input.timestamp, message=message)

    async def raw_to_text(self, raw_input: Optional[VisionData]) -> None:
        """
        Vision 데이터를 처리하고 최신 설명 업데이트.

        Parameters
        ----------
        raw_input : Optional[VisionData]
            처리할 Vision 데이터
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.latest_description = pending_message.message
            if raw_input:
                self.latest_objects = raw_input.detected_objects

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        최신 버퍼 내용을 LLM 입력 형식으로 포맷팅.

        Returns
        -------
        Optional[str]
            포맷팅된 입력 문자열, 버퍼가 비어있으면 None
        """
        if self.latest_description is None:
            return None

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{self.latest_description}
// END
"""
        # 버퍼 초기화
        self.latest_description = None
        self.latest_objects = []
        return result

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        현재 RGB 프레임 반환.

        Returns
        -------
        Optional[np.ndarray]
            현재 RGB 프레임
        """
        # TODO: camera_provider에서 현재 프레임 반환
        return None

    def get_current_depth(self) -> Optional[np.ndarray]:
        """
        현재 깊이 프레임 반환.

        Returns
        -------
        Optional[np.ndarray]
            현재 깊이 프레임
        """
        # TODO: camera_provider에서 현재 깊이 프레임 반환
        return None

    def start(self) -> None:
        """
        Vision Sensor 시작.

        모든 관련 Provider를 시작합니다.
        """
        # TODO: Provider 시작 로직
        # self.camera_provider.start()
        # if self.config.enable_segmentation:
        #     self.segmentation_provider.start()
        # if self.config.enable_bev:
        #     self.bev_provider.start()
        # if self.config.enable_pointcloud:
        #     self.pointcloud_provider.start()
        # self.vlm_provider.start()
        logging.info("VisionSensor started")

    def stop(self) -> None:
        """
        Vision Sensor 정지.

        모든 관련 Provider를 정지합니다.
        """
        # TODO: Provider 정지 로직
        # self.camera_provider.stop()
        # self.segmentation_provider.stop()
        # self.bev_provider.stop()
        # self.pointcloud_provider.stop()
        # self.vlm_provider.stop()
        logging.info("VisionSensor stopped")
