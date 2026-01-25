"""
Sound Sensor - Input Orchestrator Sound 모듈

이 모듈은 마이크 입력을 통해 음성 데이터를 수집하고,
ASR(Automatic Speech Recognition) 결과를 Fuser에 전달합니다.

Architecture:
    AudioBg -> AudioProvider -> SoundSensor -> InputOrchestrator -> Fuser

Dependencies:
    - AudioProvider: 오디오 스트림 관리
    - STTProvider: Speech-to-Text 변환
"""

import asyncio
import logging
import time
from queue import Empty, Queue
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput

# TODO: Provider import 추가 예정
# from providers.audio_provider import AudioProvider
# from providers.stt_provider import STTProvider


class SoundSensorConfig(SensorConfig):
    """
    Sound Sensor 설정.

    Parameters
    ----------
    sample_rate : int
        오디오 샘플링 레이트 (Hz)
    chunk_size : int
        오디오 청크 크기 (samples)
    channels : int
        오디오 채널 수 (1: mono, 2: stereo)
    device_id : Optional[int]
        마이크 디바이스 ID (None이면 시스템 기본값)
    device_name : Optional[str]
        마이크 디바이스 이름
    language : str
        음성 인식 언어 코드
    vad_enabled : bool
        Voice Activity Detection 활성화 여부
    vad_threshold : float
        VAD 임계값 (0.0 ~ 1.0)
    buffer_duration_ms : int
        오디오 버퍼 지속 시간 (ms)
    """

    sample_rate: int = Field(default=16000, description="오디오 샘플링 레이트 (Hz)")
    chunk_size: int = Field(default=1024, description="오디오 청크 크기")
    channels: int = Field(default=1, description="오디오 채널 수")
    device_id: Optional[int] = Field(default=None, description="마이크 디바이스 ID")
    device_name: Optional[str] = Field(default=None, description="마이크 디바이스 이름")
    language: str = Field(default="ko-KR", description="음성 인식 언어 코드")
    vad_enabled: bool = Field(default=True, description="VAD 활성화 여부")
    vad_threshold: float = Field(default=0.5, description="VAD 임계값")
    buffer_duration_ms: int = Field(default=200, description="버퍼 지속 시간 (ms)")


class SoundSensor(FuserInput[SoundSensorConfig, Optional[str]]):
    """
    Sound Sensor - 음성 입력 처리 센서.

    마이크로부터 오디오 데이터를 수집하고, STT를 통해 텍스트로 변환하여
    InputOrchestrator에 전달합니다.

    Attributes
    ----------
    descriptor_for_LLM : str
        LLM에 전달될 입력 소스 설명자
    message_buffer : Queue[str]
        STT 결과 메시지 버퍼
    messages : list[str]
        처리된 메시지 목록

    Methods
    -------
    _poll() -> Optional[str]
        버퍼에서 새 메시지 폴링
    _raw_to_text(raw_input) -> Optional[Message]
        원시 입력을 텍스트 메시지로 변환
    formatted_latest_buffer() -> Optional[str]
        최신 버퍼 내용을 포맷팅하여 반환
    """

    def __init__(self, config: SoundSensorConfig):
        """
        Sound Sensor 초기화.

        Parameters
        ----------
        config : SoundSensorConfig
            센서 설정
        """
        super().__init__(config)

        # LLM 입력 설명자
        self.descriptor_for_LLM = "Voice"

        # 메시지 버퍼
        self.message_buffer: Queue[str] = Queue()
        self.messages: list[str] = []

        # TODO: Provider 초기화
        # self.audio_provider = AudioProvider(...)
        # self.stt_provider = STTProvider(...)

        logging.info(
            f"SoundSensor initialized with language={config.language}, "
            f"sample_rate={config.sample_rate}"
        )

    def _handle_stt_result(self, text: str) -> None:
        """
        STT 결과 콜백 핸들러.

        Parameters
        ----------
        text : str
            STT로 변환된 텍스트
        """
        if text and len(text.strip()) > 0:
            self.message_buffer.put(text)
            logging.debug(f"STT result received: {text}")

    async def _poll(self) -> Optional[str]:
        """
        메시지 버퍼에서 새 메시지를 폴링.

        Returns
        -------
        Optional[str]
            버퍼에 메시지가 있으면 반환, 없으면 None
        """
        await asyncio.sleep(0.1)
        try:
            message = self.message_buffer.get_nowait()
            return message
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        원시 입력을 Message 객체로 변환.

        Parameters
        ----------
        raw_input : Optional[str]
            STT로 변환된 텍스트

        Returns
        -------
        Optional[Message]
            타임스탬프가 포함된 Message 객체
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]) -> None:
        """
        원시 입력을 처리하고 메시지 목록에 추가.

        Parameters
        ----------
        raw_input : Optional[str]
            처리할 원시 입력
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            if len(self.messages) == 0:
                self.messages.append(pending_message.message)
            else:
                # 연속된 메시지는 하나로 결합
                self.messages[-1] = f"{self.messages[-1]} {pending_message.message}"

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        최신 버퍼 내용을 LLM 입력 형식으로 포맷팅.

        Returns
        -------
        Optional[str]
            포맷팅된 입력 문자열, 버퍼가 비어있으면 None
        """
        if len(self.messages) == 0:
            return None

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{self.messages[-1]}
// END
"""
        # 버퍼 초기화
        self.messages = []
        return result

    def start(self) -> None:
        """
        Sound Sensor 시작.

        AudioProvider와 STTProvider를 시작합니다.
        """
        # TODO: Provider 시작 로직
        # self.audio_provider.start()
        # self.stt_provider.start()
        logging.info("SoundSensor started")

    def stop(self) -> None:
        """
        Sound Sensor 정지.

        AudioProvider와 STTProvider를 정지합니다.
        """
        # TODO: Provider 정지 로직
        # self.audio_provider.stop()
        # self.stt_provider.stop()
        logging.info("SoundSensor stopped")
