"""
Speaker Provider - 오디오 출력 스트림 관리

이 모듈은 스피커로 오디오 데이터를 출력합니다.

Architecture:
    TTSProvider -> SpeakerProvider -> Speaker Device

Dependencies:
    - TTSProvider: 오디오 데이터 소스

Note:
    이 Provider는 Singleton 패턴을 사용하여 시스템 전체에서
    하나의 오디오 출력 스트림만 유지합니다.
"""

import logging
import queue
import threading
from typing import Callable, List, Optional

from .singleton import singleton

# TODO: 오디오 출력 라이브러리 import
# import pyaudio
# import sounddevice as sd


@singleton
class SpeakerProvider:
    """
    Speaker Provider - 오디오 출력 스트림 관리자.

    오디오 데이터를 큐에 저장하고 스피커로 재생합니다.

    Attributes
    ----------
    running : bool
        Provider 실행 상태
    sample_rate : int
        오디오 샘플링 레이트 (Hz)
    channels : int
        오디오 채널 수
    device_id : Optional[int]
        스피커 디바이스 ID
    volume : float
        볼륨 레벨 (0.0 ~ 1.0)

    Methods
    -------
    start()
        오디오 출력 스트림 시작
    stop()
        오디오 출력 스트림 정지
    queue_audio(audio_data)
        오디오 데이터 큐에 추가
    play_audio(audio_data)
        오디오 데이터 즉시 재생
    stop_playback()
        현재 재생 중단
    set_volume(volume)
        볼륨 설정
    register_playback_callback(callback)
        재생 상태 콜백 등록
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        device_id: Optional[int] = None,
        device_name: Optional[str] = None,
        volume: float = 1.0,
        buffer_size: int = 4096,
    ):
        """
        Speaker Provider 초기화.

        Parameters
        ----------
        sample_rate : int
            오디오 샘플링 레이트 (Hz), 기본값 44100
        channels : int
            오디오 채널 수, 기본값 1 (mono)
        device_id : Optional[int]
            스피커 디바이스 ID, None이면 시스템 기본값
        device_name : Optional[str]
            스피커 디바이스 이름
        volume : float
            초기 볼륨 레벨 (0.0 ~ 1.0), 기본값 1.0
        buffer_size : int
            오디오 버퍼 크기, 기본값 4096
        """
        self.running: bool = False
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_id = device_id
        self.device_name = device_name
        self._volume = max(0.0, min(1.0, volume))
        self.buffer_size = buffer_size

        # 오디오 큐
        self._audio_queue: queue.Queue[bytes] = queue.Queue()

        # 콜백 리스트
        self._playback_callbacks: List[Callable[[str], None]] = []

        # 상태 변수
        self._is_playing: bool = False
        self._should_stop: bool = False
        self._lock = threading.Lock()
        self._playback_thread: Optional[threading.Thread] = None

        logging.info(
            f"SpeakerProvider initialized: rate={sample_rate}, "
            f"channels={channels}, device={device_id or device_name or 'default'}"
        )

    @property
    def volume(self) -> float:
        """현재 볼륨 레벨 반환."""
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        """볼륨 레벨 설정."""
        self._volume = max(0.0, min(1.0, value))
        logging.debug(f"Volume set to {self._volume}")

    def register_playback_callback(self, callback: Callable[[str], None]) -> None:
        """
        재생 상태 콜백 등록.

        Parameters
        ----------
        callback : Callable[[str], None]
            재생 상태를 받을 콜백 함수
            상태: "started", "playing", "paused", "stopped", "completed"
        """
        if callback not in self._playback_callbacks:
            self._playback_callbacks.append(callback)
            logging.debug(f"Playback callback registered: {callback.__name__}")

    def unregister_playback_callback(self, callback: Callable[[str], None]) -> None:
        """
        재생 상태 콜백 해제.

        Parameters
        ----------
        callback : Callable[[str], None]
            해제할 콜백 함수
        """
        if callback in self._playback_callbacks:
            self._playback_callbacks.remove(callback)
            logging.debug(f"Playback callback unregistered: {callback.__name__}")

    def _notify_playback_state(self, state: str) -> None:
        """
        재생 상태 변경 알림.

        Parameters
        ----------
        state : str
            새 재생 상태
        """
        for callback in self._playback_callbacks:
            try:
                callback(state)
            except Exception as e:
                logging.error(f"Playback callback error: {e}")

    def queue_audio(self, audio_data: bytes) -> None:
        """
        오디오 데이터를 재생 큐에 추가.

        Parameters
        ----------
        audio_data : bytes
            재생할 오디오 데이터
        """
        if not self.running:
            logging.warning("SpeakerProvider is not running. Call start() first.")
            return

        self._audio_queue.put(audio_data)
        logging.debug(f"Audio queued: {len(audio_data)} bytes")

    def play_audio(self, audio_data: bytes, blocking: bool = False) -> None:
        """
        오디오 데이터 즉시 재생.

        Parameters
        ----------
        audio_data : bytes
            재생할 오디오 데이터
        blocking : bool
            True이면 재생 완료까지 블로킹, 기본값 False
        """
        if not self.running:
            logging.warning("SpeakerProvider is not running. Call start() first.")
            return

        # 큐 비우기
        self.clear_queue()

        # 오디오 큐에 추가
        self._audio_queue.put(audio_data)

        if blocking:
            # 재생 완료 대기
            self._audio_queue.join()

    def clear_queue(self) -> None:
        """
        재생 큐 비우기.
        """
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
            except queue.Empty:
                break
        logging.debug("Audio queue cleared")

    def stop_playback(self) -> None:
        """
        현재 재생 중단.
        """
        with self._lock:
            self._should_stop = True
            self.clear_queue()
        self._notify_playback_state("stopped")
        logging.info("Playback stopped")

    def _playback_loop(self) -> None:
        """
        재생 루프 (별도 스레드에서 실행).
        """
        while self.running:
            try:
                # 큐에서 오디오 데이터 가져오기 (1초 타임아웃)
                audio_data = self._audio_queue.get(timeout=1.0)

                if self._should_stop:
                    self._should_stop = False
                    self._audio_queue.task_done()
                    continue

                # 재생 시작 알림
                with self._lock:
                    self._is_playing = True
                self._notify_playback_state("playing")

                # TODO: 실제 오디오 재생 구현
                # 볼륨 적용 및 스피커 출력
                # self._play_audio_chunk(audio_data)

                # 재생 완료
                with self._lock:
                    self._is_playing = False
                self._audio_queue.task_done()

            except queue.Empty:
                # 타임아웃 - 계속 대기
                continue
            except Exception as e:
                logging.error(f"Playback error: {e}")
                with self._lock:
                    self._is_playing = False

        self._notify_playback_state("stopped")

    def is_playing(self) -> bool:
        """
        현재 재생 중인지 확인.

        Returns
        -------
        bool
            재생 중이면 True
        """
        with self._lock:
            return self._is_playing

    def get_queue_size(self) -> int:
        """
        재생 큐 크기 반환.

        Returns
        -------
        int
            큐에 대기 중인 오디오 청크 수
        """
        return self._audio_queue.qsize()

    def start(self) -> None:
        """
        오디오 출력 스트림 시작.
        """
        if self.running:
            logging.warning("SpeakerProvider is already running")
            return

        self.running = True
        self._should_stop = False

        # 재생 스레드 시작
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True
        )
        self._playback_thread.start()

        self._notify_playback_state("started")
        logging.info("SpeakerProvider started")

    def stop(self) -> None:
        """
        오디오 출력 스트림 정지.
        """
        if not self.running:
            logging.warning("SpeakerProvider is not running")
            return

        self.running = False
        self._should_stop = True
        self.clear_queue()

        # 재생 스레드 종료 대기
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)

        self._notify_playback_state("stopped")
        logging.info("SpeakerProvider stopped")

    def configure(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        device_id: Optional[int] = None,
        volume: Optional[float] = None,
    ) -> None:
        """
        Provider 설정 변경.

        Parameters
        ----------
        sample_rate : Optional[int]
            새 샘플링 레이트
        channels : Optional[int]
            새 채널 수
        device_id : Optional[int]
            새 디바이스 ID
        volume : Optional[float]
            새 볼륨 레벨
        """
        restart_needed = False

        if sample_rate is not None and sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            restart_needed = True

        if channels is not None and channels != self.channels:
            self.channels = channels
            restart_needed = True

        if device_id is not None and device_id != self.device_id:
            self.device_id = device_id
            restart_needed = True

        if volume is not None:
            self.volume = volume

        if restart_needed and self.running:
            self.stop()
            self.start()
            logging.info("SpeakerProvider reconfigured and restarted")
