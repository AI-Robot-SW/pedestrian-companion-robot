"""
Audio Provider - 오디오 입력 스트림 관리

이 모듈은 마이크로부터 오디오 데이터를 캡처하고,
VAD(Voice Activity Detection)를 통해 음성 구간을 감지합니다.

Architecture:
    AudioBg -> AudioProvider -> Microphone Device
                    ↓
              STTProvider (오디오 데이터 전달)

Dependencies:
    - om1_speech (AudioInputStream): 오디오 스트림 처리
    - STTProvider: 음성-텍스트 변환

Note:
    이 Provider는 Singleton 패턴을 사용하여 시스템 전체에서
    하나의 오디오 입력 스트림만 유지합니다.
"""

import logging
import threading
from typing import Callable, List, Optional

from singleton import singleton

# /home/nvidia/pedestrian-companion-robot/src/om1_speech.py 경로에 파일모듈로 존재
from om1_speech import AudioInputStream, SileroVAD


@singleton
class AudioProvider:
    """
    Audio Provider - 오디오 입력 스트림 관리자.

    마이크로부터 오디오 데이터를 캡처하고, VAD를 통해 음성 구간을 감지하며,
    등록된 콜백으로 오디오 데이터를 전달합니다.

    Attributes
    ----------
    running : bool
        Provider 실행 상태
    sample_rate : int
        오디오 샘플링 레이트 (Hz)
    chunk_size : int
        오디오 청크 크기 (samples)
    channels : int
        오디오 채널 수
    device_id : Optional[int]
        마이크 디바이스 ID
    device_name : Optional[str]
        마이크 디바이스 이름
    vad_enabled : bool
        VAD 활성화 여부
    vad_threshold : float
        VAD 임계값
    vad_min_silence_duration_ms : int
        VAD 최소 무음 길이 (ms)
    vad_speech_pad_ms : int
        VAD 양쪽 패딩 (ms)
    vad_min_speech_duration_ms : int
        VAD 최소 발화 길이 (ms)

    Methods
    -------
    start()
        오디오 스트림 시작
    stop()
        오디오 스트림 정지
    register_audio_callback(callback)
        오디오 데이터 콜백 등록
    register_vad_callback(callback)
        VAD 이벤트 콜백 등록
    get_audio_level() -> float
        현재 오디오 레벨 반환
    is_voice_active() -> bool
        현재 음성 활성 상태 반환
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device_id: Optional[int] = None,
        device_name: Optional[str] = None,
        vad_enabled: bool = True,
        vad_threshold: float = 0.5,
        vad_min_silence_duration_ms: int = 100,
        vad_speech_pad_ms: int = 30,
        vad_min_speech_duration_ms: int = 250,
        vad_device: str = "cpu",
        buffer_duration_ms: int = 200,
    ):
        """
        Audio Provider 초기화.

        Parameters
        ----------
        sample_rate : int
            오디오 샘플링 레이트 (Hz), 기본값 16000
        chunk_size : int
            오디오 청크 크기 (samples), 기본값 1024
        channels : int
            오디오 채널 수, 기본값 1 (mono)
        device_id : Optional[int]
            마이크 디바이스 ID, None이면 시스템 기본값
        device_name : Optional[str]
            마이크 디바이스 이름
        vad_enabled : bool
            VAD 활성화 여부, 기본값 True
        vad_threshold : float
            VAD 임계값 (0.0 ~ 1.0), 기본값 0.5
        vad_min_silence_duration_ms : int
            VAD 최소 무음 길이 (ms), 기본값 100
        vad_speech_pad_ms : int
            VAD 양쪽 패딩 (ms), 기본값 30
        vad_min_speech_duration_ms : int
            VAD 최소 발화 길이 (ms), 기본값 250
        vad_device : str
            VAD 디바이스 ("cpu" 또는 "cuda")
        buffer_duration_ms : int
            오디오 버퍼 지속 시간 (ms), 기본값 200
        """
        self.running: bool = False
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_id = device_id
        self.device_name = device_name
        self.vad_enabled = vad_enabled
        self.vad_threshold = vad_threshold
        self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self.vad_speech_pad_ms = vad_speech_pad_ms
        self.vad_min_speech_duration_ms = vad_min_speech_duration_ms
        self.vad_device = vad_device
        self.buffer_duration_ms = buffer_duration_ms

        # 콜백 리스트
        self._audio_callbacks: List[Callable[..., None]] = []
        self._vad_callbacks: List[Callable[[bool], None]] = []

        # 상태 변수
        self._current_audio_level: float = 0.0
        self._is_voice_active: bool = False
        self._lock = threading.Lock()

        # AudioInputStream 초기화, provider.on_audio_data 콜백 등록
        self._audio_stream: AudioInputStream = AudioInputStream(
            rate=sample_rate,
            chunk=chunk_size,
            device=device_id,
            device_name=device_name,
            audio_data_callback=self._on_audio_data,
        )
        self._vad_engine: Optional[SileroVAD] = None
        self._init_vad_engine()

        logging.info(
            f"AudioProvider initialized: rate={sample_rate}, "
            f"chunk={chunk_size}, device={device_id or device_name or 'default'}"
        )

    def _init_vad_engine(self) -> None:
        if not self.vad_enabled:
            self._vad_engine = None
            return
        try:
            self._vad_engine = SileroVAD(
                sampling_rate=self.sample_rate,
                threshold=self.vad_threshold,
                min_silence_duration_ms=self.vad_min_silence_duration_ms,
                speech_pad_ms=self.vad_speech_pad_ms,
                min_speech_duration_ms=self.vad_min_speech_duration_ms,
                chunk_size=self.chunk_size,
                device=self.vad_device,
            )
        except Exception as e:
            logging.error(f"SileroVAD init failed: {e}")
            self._vad_engine = None
            self.vad_enabled = False

    def register_audio_callback(self, callback: Callable[..., None]) -> None:
        """
        오디오 데이터 콜백 등록.

        Parameters
        ----------
        callback : Callable[[bytes], None]
            오디오 청크를 받을 콜백 함수
        """
        if callback not in self._audio_callbacks:
            self._audio_callbacks.append(callback)
            logging.debug(f"Audio callback registered: {callback.__name__}")

    def unregister_audio_callback(self, callback: Callable[..., None]) -> None:
        """
        오디오 데이터 콜백 해제.

        Parameters
        ----------
        callback : Callable[[bytes], None]
            해제할 콜백 함수
        """
        if callback in self._audio_callbacks:
            self._audio_callbacks.remove(callback)
            logging.debug(f"Audio callback unregistered: {callback.__name__}")

    def register_vad_callback(self, callback: Callable[[bool], None]) -> None:
        """
        VAD 이벤트 콜백 등록.

        Parameters
        ----------
        callback : Callable[[bool], None]
            음성 활성 상태 변경 시 호출될 콜백 (True: 음성 시작, False: 음성 종료)
        """
        if callback not in self._vad_callbacks:
            self._vad_callbacks.append(callback)
            logging.debug(f"VAD callback registered: {callback.__name__}")

    def unregister_vad_callback(self, callback: Callable[[bool], None]) -> None:
        """
        VAD 이벤트 콜백 해제.

        Parameters
        ----------
        callback : Callable[[bool], None]
            해제할 콜백 함수
        """
        if callback in self._vad_callbacks:
            self._vad_callbacks.remove(callback)
            logging.debug(f"VAD callback unregistered: {callback.__name__}")

    def _on_audio_data(
        self, audio_chunk: bytes, frame_count: int, time_info: dict, status_flags: int
    ) -> None:
        """
        오디오 데이터 수신 내부 핸들러.

        Parameters
        ----------
        audio_chunk : bytes
            수신된 오디오 청크
        """
        # 오디오 레벨 계산
        self._update_audio_level(audio_chunk)

        # VAD 처리
        if self.vad_enabled:
            self._process_vad(audio_chunk)

        # 등록된 콜백 호출
        for callback in self._audio_callbacks:
            try:
                callback(audio_chunk, frame_count, time_info, status_flags)
            except Exception as e:
                logging.error(f"Audio callback error: {e}")

    def _update_audio_level(self, audio_chunk: bytes) -> None:
        """
        오디오 레벨 업데이트.

        Parameters
        ----------
        audio_chunk : bytes
            오디오 청크
        """
        if not audio_chunk:
            return
        # PCM 16-bit signed little-endian RMS -> 0~1 사이의 값으로 정규화 (디버깅용)
        try:
            import math
            import struct

            sample_count = len(audio_chunk) // 2
            if sample_count == 0:
                return
            samples = struct.unpack("<" + "h" * sample_count, audio_chunk)
            rms = math.sqrt(sum(s * s for s in samples) / sample_count)
            level = min(1.0, rms / 32768.0)
            with self._lock:
                self._current_audio_level = level
        except Exception as e:
            logging.error(f"Audio level update failed: {e}")

    def _process_vad(self, audio_chunk: bytes) -> None:
        """
        VAD 처리.

        Parameters
        ----------
        audio_chunk : bytes
            오디오 청크
        """
        if not self.vad_enabled or self._vad_engine is None:
            return
        try:
            self._vad_engine.process(audio_chunk)
            is_voice = self._vad_engine.speech_active
        except Exception as e:
            logging.error(f"VAD processing error: {e}")
            return

        with self._lock:
            was_active = self._is_voice_active
            self._is_voice_active = is_voice

        if was_active != is_voice:
            for callback in list(self._vad_callbacks):
                try:
                    callback(is_voice)
                except Exception as e:
                    logging.error(f"VAD callback error: {e}")

    def get_audio_level(self) -> float:
        """
        현재 오디오 레벨 반환.

        Returns
        -------
        float
            현재 오디오 레벨 (0.0 ~ 1.0)
        """
        with self._lock:
            return self._current_audio_level

    def is_voice_active(self) -> bool:
        """
        현재 음성 활성 상태 반환.

        Returns
        -------
        bool
            음성이 감지되면 True
        """
        with self._lock:
            return self._is_voice_active

    def start(self) -> None:
        """
        오디오 스트림 시작.
        """
        if self.running:
            logging.warning("AudioProvider is already running")
            return

        self.running = True
        self._audio_stream.start()
        logging.info("AudioProvider started")

    def stop(self) -> None:
        """
        오디오 스트림 정지.
        """
        if not self.running:
            logging.warning("AudioProvider is not running")
            return

        self.running = False
        self._audio_stream.stop()
        logging.info("AudioProvider stopped")

    def configure(
        self,
        sample_rate: Optional[int] = None,
        chunk_size: Optional[int] = None,
        device_id: Optional[int] = None,
        device_name: Optional[str] = None,
        vad_enabled: Optional[bool] = None,
        vad_threshold: Optional[float] = None,
        vad_min_silence_duration_ms: Optional[int] = None,
        vad_speech_pad_ms: Optional[int] = None,
        vad_min_speech_duration_ms: Optional[int] = None,
        vad_device: Optional[str] = None,
    ) -> None:
        """
        Provider 설정 변경.

        변경된 설정이 있으면 스트림을 재시작합니다.

        Parameters
        ----------
        sample_rate : Optional[int]
            새 샘플링 레이트
        chunk_size : Optional[int]
            새 청크 크기
        device_id : Optional[int]
            새 디바이스 ID
        device_name : Optional[str]
            새 디바이스 이름
        vad_enabled : Optional[bool]
            VAD 활성화 여부
        vad_threshold : Optional[float]
            새 VAD 임계값
        vad_min_silence_duration_ms : Optional[int]
            새 VAD 최소 무음 길이 (ms)
        vad_speech_pad_ms : Optional[int]
            새 VAD 양쪽 패딩 (ms)
        vad_min_speech_duration_ms : Optional[int]
            새 VAD 최소 발화 길이 (ms)
        vad_device : Optional[str]
            새 VAD 디바이스
        """
        restart_needed = False
        vad_reinit_needed = False

        if sample_rate is not None and sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            restart_needed = True
            vad_reinit_needed = True

        if chunk_size is not None and chunk_size != self.chunk_size:
            self.chunk_size = chunk_size
            restart_needed = True
            vad_reinit_needed = True

        if device_id is not None and device_id != self.device_id:
            self.device_id = device_id
            restart_needed = True

        if device_name is not None and device_name != self.device_name:
            self.device_name = device_name
            restart_needed = True

        if vad_enabled is not None and vad_enabled != self.vad_enabled:
            self.vad_enabled = vad_enabled
            vad_reinit_needed = True

        if vad_threshold is not None and vad_threshold != self.vad_threshold:
            self.vad_threshold = vad_threshold
            vad_reinit_needed = True

        if (
            vad_min_silence_duration_ms is not None
            and vad_min_silence_duration_ms != self.vad_min_silence_duration_ms
        ):
            self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
            vad_reinit_needed = True

        if vad_speech_pad_ms is not None and vad_speech_pad_ms != self.vad_speech_pad_ms:
            self.vad_speech_pad_ms = vad_speech_pad_ms
            vad_reinit_needed = True

        if (
            vad_min_speech_duration_ms is not None
            and vad_min_speech_duration_ms != self.vad_min_speech_duration_ms
        ):
            self.vad_min_speech_duration_ms = vad_min_speech_duration_ms
            vad_reinit_needed = True

        if vad_device is not None and vad_device != self.vad_device:
            self.vad_device = vad_device
            vad_reinit_needed = True

        if vad_reinit_needed:
            if self.vad_enabled:
                self._init_vad_engine()
            else:
                self._vad_engine = None
                with self._lock:
                    self._is_voice_active = False

        if restart_needed and self.running:
            self.stop()
            # TODO: 새 설정으로 AudioInputStream 재생성
            self.start()
            logging.info("AudioProvider reconfigured and restarted")
