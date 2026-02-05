"""
STT Provider - Speech-to-Text 변환 관리

이 모듈은 오디오 데이터를 받아 Google Cloud STT API를 통해
텍스트로 변환합니다.

Architecture:
    AudioProvider -> STTProvider -> Google Cloud STT API (Streaming)
                          ↓
                    SoundSensor (결과 콜백)

Dependencies:
    - google-cloud-speech: Google Cloud Speech-to-Text 클라이언트
    - AudioProvider: 오디오 데이터 소스

Note:
    이 Provider는 Singleton 패턴을 사용합니다.
    Google Cloud STT Streaming API를 사용하여 실시간 음성 인식을 수행합니다.

Google Cloud STT API:
    - Streaming Recognition: 실시간 오디오 스트림 처리
    - 지원 인코딩: LINEAR16 (PCM), FLAC, MULAW, AMR 등
    - 지원 샘플레이트: 8000Hz ~ 48000Hz
    - 중간 결과(interim results) 지원

Environment Variables:
    - GOOGLE_APPLICATION_CREDENTIALS: Google Cloud 서비스 계정 JSON 파일 경로
    - OM_API_KEY: OpenMind API 키 (WebSocket 백엔드용)
"""

import json
import logging
import os
import threading
from enum import Enum
from typing import Callable, Dict, List, Optional

from .singleton import singleton

# TODO: Google Cloud Speech import 추가 예정
# from google.cloud import speech
# from google.cloud.speech import RecognitionConfig, StreamingRecognitionConfig


def _get_env_or_default(env_var: str, default: Optional[str] = None) -> Optional[str]:
    """환경 변수에서 값을 가져오거나 기본값 반환."""
    return os.environ.get(env_var, default)


class STTBackend(Enum):
    """STT 백엔드 종류."""

    GOOGLE_CLOUD = "google_cloud"
    GOOGLE_WEBSOCKET = "google_websocket"  # OpenMind 프록시 WebSocket
    LOCAL = "local"


class AudioEncoding(Enum):
    """오디오 인코딩 형식 (Google Cloud STT 지원)."""

    LINEAR16 = "LINEAR16"  # PCM 16-bit signed little-endian
    FLAC = "FLAC"
    MULAW = "MULAW"
    AMR = "AMR"
    AMR_WB = "AMR_WB"
    OGG_OPUS = "OGG_OPUS"
    SPEEX_WITH_HEADER_BYTE = "SPEEX_WITH_HEADER_BYTE"
    WEBM_OPUS = "WEBM_OPUS"


# 지원 언어 코드 매핑 (Google Cloud STT BCP-47 코드)
LANGUAGE_CODE_MAP: Dict[str, str] = {
    "english": "en-US",
    "english_uk": "en-GB",
    "korean": "ko-KR",
    "chinese": "cmn-Hans-CN",
    "chinese_traditional": "cmn-Hant-TW",
    "japanese": "ja-JP",
    "german": "de-DE",
    "french": "fr-FR",
    "spanish": "es-ES",
    "italian": "it-IT",
    "portuguese": "pt-BR",
    "russian": "ru-RU",
    "arabic": "ar-SA",
}

# Google Cloud STT 기본 설정
GOOGLE_CLOUD_STT_DEFAULT_CONFIG = {
    "encoding": AudioEncoding.LINEAR16.value,
    "sample_rate_hertz": 16000,
    "language_code": "ko-KR",
    "enable_automatic_punctuation": True,
    "model": "latest_long",  # 또는 "latest_short", "phone_call", "video", "default"
    "use_enhanced": True,  # Enhanced 모델 사용 (더 정확하지만 비용 증가)
}


@singleton
class STTProvider:
    """
    STT Provider - Speech-to-Text 변환 관리자.

    오디오 데이터를 Google Cloud STT API로 전송하고, 변환된 텍스트를
    등록된 콜백으로 전달합니다.

    Attributes
    ----------
    running : bool
        Provider 실행 상태
    backend : STTBackend
        사용 중인 STT 백엔드
    language_code : str
        음성 인식 언어 코드 (BCP-47)
    enable_interim_results : bool
        중간 결과 활성화 여부

    Methods
    -------
    start()
        STT 서비스 연결 시작
    stop()
        STT 서비스 연결 종료
    send_audio(audio_chunk)
        오디오 데이터 전송
    register_result_callback(callback)
        STT 결과 콜백 등록
    register_interim_callback(callback)
        중간 결과 콜백 등록
    end_utterance()
        발화 종료 신호 전송

    Google Cloud STT Features
    -------------------------
    - Streaming Recognition: 실시간 스트리밍 음성 인식
    - Interim Results: 중간 결과 반환
    - Automatic Punctuation: 자동 구두점 삽입
    - Enhanced Models: 향상된 음성 인식 모델
    - Speaker Diarization: 화자 분리 (선택적)
    """

    def __init__(
        self,
        backend: STTBackend = STTBackend.GOOGLE_CLOUD,
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None,
        credentials_path: Optional[str] = None,
        language: str = "korean",
        enable_interim_results: bool = True,
        sample_rate: int = 16000,
        encoding: str = "LINEAR16",
        # Google Cloud STT 전용 파라미터
        model: str = "latest_long",
        use_enhanced: bool = True,
        enable_automatic_punctuation: bool = True,
        enable_speaker_diarization: bool = False,
        diarization_speaker_count: int = 2,
        single_utterance: bool = False,
        speech_contexts: Optional[List[Dict]] = None,
    ):
        """
        STT Provider 초기화.

        Parameters
        ----------
        backend : STTBackend
            STT 백엔드 종류, 기본값 GOOGLE_CLOUD
        ws_url : Optional[str]
            WebSocket URL (GOOGLE_WEBSOCKET 백엔드용)
        api_key : Optional[str]
            API 키 (WebSocket 백엔드용)
        credentials_path : Optional[str]
            Google Cloud 서비스 계정 JSON 파일 경로
        language : str
            음성 인식 언어, 기본값 "korean"
        enable_interim_results : bool
            중간 결과 활성화 여부, 기본값 True
        sample_rate : int
            오디오 샘플링 레이트, 기본값 16000
        encoding : str
            오디오 인코딩 형식, 기본값 "LINEAR16"

        Google Cloud STT 전용 Parameters
        --------------------------------
        model : str
            음성 인식 모델
            - "latest_long": 긴 오디오용 최신 모델
            - "latest_short": 짧은 오디오용 최신 모델
            - "phone_call": 전화 통화 최적화
            - "video": 비디오 오디오 최적화
            - "default": 기본 모델
        use_enhanced : bool
            Enhanced 모델 사용 여부 (더 정확, 비용 증가)
        enable_automatic_punctuation : bool
            자동 구두점 삽입 활성화
        enable_speaker_diarization : bool
            화자 분리 활성화
        diarization_speaker_count : int
            예상 화자 수 (화자 분리 시)
        single_utterance : bool
            단일 발화 모드 (발화 종료 시 자동 종료)
        speech_contexts : Optional[List[Dict]]
            음성 컨텍스트 힌트 (특정 단어/구문 인식률 향상)
            예: [{"phrases": ["OpenMind", "KIST"], "boost": 20}]
        """
        self.running: bool = False
        self.backend = backend
        self.ws_url = ws_url or "wss://api.openmind.org/api/core/google/asr"

        # API 키 및 인증 정보는 환경 변수에서 로드 (파라미터 우선)
        self.api_key = api_key or _get_env_or_default("OM_API_KEY")
        self.credentials_path = credentials_path or _get_env_or_default(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        self.enable_interim_results = enable_interim_results
        self.sample_rate = sample_rate
        self.encoding = encoding

        # Google Cloud STT 전용 설정
        self._model = model
        self._use_enhanced = use_enhanced
        self._enable_automatic_punctuation = enable_automatic_punctuation
        self._enable_speaker_diarization = enable_speaker_diarization
        self._diarization_speaker_count = diarization_speaker_count
        self._single_utterance = single_utterance
        self._speech_contexts = speech_contexts or []

        # 언어 코드 설정
        language_lower = language.strip().lower()
        if language_lower not in LANGUAGE_CODE_MAP:
            logging.warning(
                f"Language '{language}' not supported. "
                f"Supported: {list(LANGUAGE_CODE_MAP.keys())}. Defaulting to Korean."
            )
            language_lower = "korean"
        self.language_code = LANGUAGE_CODE_MAP[language_lower]

        # 콜백 리스트
        self._result_callbacks: List[Callable[[str], None]] = []
        self._interim_callbacks: List[Callable[[str], None]] = []

        # 상태 변수
        self._is_listening: bool = False
        self._current_transcript: str = ""
        self._lock = threading.Lock()

        # Google Cloud Speech 클라이언트
        self._speech_client = None
        self._streaming_config = None

        # TODO: Google Cloud Speech 클라이언트 초기화
        # if backend == STTBackend.GOOGLE_CLOUD:
        #     if credentials_path:
        #         self._speech_client = speech.SpeechClient.from_service_account_json(
        #             credentials_path
        #         )
        #     else:
        #         self._speech_client = speech.SpeechClient()
        #
        #     recognition_config = speech.RecognitionConfig(
        #         encoding=speech.RecognitionConfig.AudioEncoding[encoding],
        #         sample_rate_hertz=sample_rate,
        #         language_code=self.language_code,
        #         enable_automatic_punctuation=enable_automatic_punctuation,
        #         model=model,
        #         use_enhanced=use_enhanced,
        #     )
        #
        #     if enable_speaker_diarization:
        #         recognition_config.diarization_config = speech.SpeakerDiarizationConfig(
        #             enable_speaker_diarization=True,
        #             min_speaker_count=1,
        #             max_speaker_count=diarization_speaker_count,
        #         )
        #
        #     if speech_contexts:
        #         recognition_config.speech_contexts = [
        #             speech.SpeechContext(phrases=ctx.get("phrases", []), boost=ctx.get("boost", 0))
        #             for ctx in speech_contexts
        #         ]
        #
        #     self._streaming_config = speech.StreamingRecognitionConfig(
        #         config=recognition_config,
        #         interim_results=enable_interim_results,
        #         single_utterance=single_utterance,
        #     )

        logging.info(
            f"STTProvider initialized: backend={backend.value}, "
            f"language={self.language_code}, model={model}, "
            f"sample_rate={sample_rate}"
        )

    def register_result_callback(self, callback: Callable[[str], None]) -> None:
        """
        STT 최종 결과 콜백 등록.

        Parameters
        ----------
        callback : Callable[[str], None]
            최종 변환 텍스트를 받을 콜백 함수
        """
        if callback not in self._result_callbacks:
            self._result_callbacks.append(callback)
            logging.debug(f"STT result callback registered: {callback.__name__}")

    def unregister_result_callback(self, callback: Callable[[str], None]) -> None:
        """
        STT 최종 결과 콜백 해제.

        Parameters
        ----------
        callback : Callable[[str], None]
            해제할 콜백 함수
        """
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)
            logging.debug(f"STT result callback unregistered: {callback.__name__}")

    def register_interim_callback(self, callback: Callable[[str], None]) -> None:
        """
        STT 중간 결과 콜백 등록.

        Parameters
        ----------
        callback : Callable[[str], None]
            중간 변환 텍스트를 받을 콜백 함수
        """
        if callback not in self._interim_callbacks:
            self._interim_callbacks.append(callback)
            logging.debug(f"STT interim callback registered: {callback.__name__}")

    def unregister_interim_callback(self, callback: Callable[[str], None]) -> None:
        """
        STT 중간 결과 콜백 해제.

        Parameters
        ----------
        callback : Callable[[str], None]
            해제할 콜백 함수
        """
        if callback in self._interim_callbacks:
            self._interim_callbacks.remove(callback)
            logging.debug(f"STT interim callback unregistered: {callback.__name__}")

    def _on_message(self, raw_message: str) -> None:
        """
        WebSocket 메시지 수신 핸들러.

        Parameters
        ----------
        raw_message : str
            수신된 JSON 메시지
        """
        try:
            message = json.loads(raw_message)

            # ASR 응답 처리
            if "asr_reply" in message:
                transcript = message["asr_reply"]
                is_final = message.get("is_final", True)

                if is_final:
                    # 최종 결과
                    self._current_transcript = transcript
                    for callback in self._result_callbacks:
                        try:
                            callback(transcript)
                        except Exception as e:
                            logging.error(f"STT result callback error: {e}")
                else:
                    # 중간 결과
                    if self.enable_interim_results:
                        for callback in self._interim_callbacks:
                            try:
                                callback(transcript)
                            except Exception as e:
                                logging.error(f"STT interim callback error: {e}")

            # 에러 처리
            elif "error" in message:
                logging.error(f"STT service error: {message['error']}")

        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON message from STT service: {e}")
        except Exception as e:
            logging.error(f"Error processing STT message: {e}")

    def send_audio(self, audio_chunk: bytes) -> None:
        """
        오디오 데이터를 ASR 서비스로 전송.

        Parameters
        ----------
        audio_chunk : bytes
            전송할 오디오 청크
        """
        if not self.running:
            logging.warning("STTProvider is not running. Call start() first.")
            return

        # TODO: WebSocket으로 오디오 전송
        # self._ws_client.send_message(audio_chunk)

    def end_utterance(self) -> None:
        """
        발화 종료 신호 전송.

        현재 발화가 끝났음을 ASR 서비스에 알립니다.
        """
        if not self.running:
            return

        # TODO: 발화 종료 메시지 전송
        # end_message = json.dumps({"action": "end_utterance"})
        # self._ws_client.send_message(end_message)
        logging.debug("End utterance signal sent")

    def get_current_transcript(self) -> str:
        """
        현재 변환된 텍스트 반환.

        Returns
        -------
        str
            현재까지 변환된 텍스트
        """
        with self._lock:
            return self._current_transcript

    def is_listening(self) -> bool:
        """
        현재 청취 상태 반환.

        Returns
        -------
        bool
            청취 중이면 True
        """
        with self._lock:
            return self._is_listening

    def start(self) -> None:
        """
        STT 서비스 연결 시작.
        """
        if self.running:
            logging.warning("STTProvider is already running")
            return

        self.running = True
        self._is_listening = True

        # TODO: WebSocket 연결 시작
        # self._ws_client.start()

        # 초기 설정 메시지 전송
        # config_message = json.dumps({
        #     "config": {
        #         "language_code": self.language_code,
        #         "sample_rate": self.sample_rate,
        #         "encoding": self.encoding,
        #         "interim_results": self.enable_interim_results,
        #     }
        # })
        # self._ws_client.send_message(config_message)

        logging.info(f"STTProvider started with language={self.language_code}")

    def stop(self) -> None:
        """
        STT 서비스 연결 종료.
        """
        if not self.running:
            logging.warning("STTProvider is not running")
            return

        self.running = False
        self._is_listening = False

        # TODO: WebSocket 연결 종료
        # self._ws_client.stop()

        logging.info("STTProvider stopped")

    def configure(
        self,
        language: Optional[str] = None,
        enable_interim_results: Optional[bool] = None,
        sample_rate: Optional[int] = None,
        # Google Cloud STT 전용 파라미터
        model: Optional[str] = None,
        use_enhanced: Optional[bool] = None,
        enable_automatic_punctuation: Optional[bool] = None,
        enable_speaker_diarization: Optional[bool] = None,
        diarization_speaker_count: Optional[int] = None,
        single_utterance: Optional[bool] = None,
        speech_contexts: Optional[List[Dict]] = None,
    ) -> None:
        """
        Provider 설정 변경.

        Parameters
        ----------
        language : Optional[str]
            새 언어 설정
        enable_interim_results : Optional[bool]
            중간 결과 활성화 여부
        sample_rate : Optional[int]
            새 샘플링 레이트

        Google Cloud STT 전용 Parameters
        --------------------------------
        model : Optional[str]
            음성 인식 모델
        use_enhanced : Optional[bool]
            Enhanced 모델 사용 여부
        enable_automatic_punctuation : Optional[bool]
            자동 구두점 삽입 활성화
        enable_speaker_diarization : Optional[bool]
            화자 분리 활성화
        diarization_speaker_count : Optional[int]
            예상 화자 수
        single_utterance : Optional[bool]
            단일 발화 모드
        speech_contexts : Optional[List[Dict]]
            음성 컨텍스트 힌트
        """
        restart_needed = False

        if language is not None:
            language_lower = language.strip().lower()
            if language_lower in LANGUAGE_CODE_MAP:
                new_code = LANGUAGE_CODE_MAP[language_lower]
                if new_code != self.language_code:
                    self.language_code = new_code
                    restart_needed = True

        if enable_interim_results is not None:
            self.enable_interim_results = enable_interim_results
            restart_needed = True

        if sample_rate is not None and sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            restart_needed = True

        # Google Cloud STT 전용 설정
        if model is not None and model != self._model:
            self._model = model
            restart_needed = True

        if use_enhanced is not None:
            self._use_enhanced = use_enhanced
            restart_needed = True

        if enable_automatic_punctuation is not None:
            self._enable_automatic_punctuation = enable_automatic_punctuation
            restart_needed = True

        if enable_speaker_diarization is not None:
            self._enable_speaker_diarization = enable_speaker_diarization
            restart_needed = True

        if diarization_speaker_count is not None:
            self._diarization_speaker_count = diarization_speaker_count
            restart_needed = True

        if single_utterance is not None:
            self._single_utterance = single_utterance
            restart_needed = True

        if speech_contexts is not None:
            self._speech_contexts = speech_contexts
            restart_needed = True

        if restart_needed and self.running:
            self.stop()
            self.start()
            logging.info("STTProvider reconfigured and restarted")

    def get_recognition_config(self) -> Dict:
        """
        현재 Google Cloud STT 설정 반환.

        Returns
        -------
        Dict
            현재 설정 딕셔너리
        """
        return {
            "backend": self.backend.value,
            "language_code": self.language_code,
            "sample_rate_hertz": self.sample_rate,
            "encoding": self.encoding,
            "model": self._model,
            "use_enhanced": self._use_enhanced,
            "enable_automatic_punctuation": self._enable_automatic_punctuation,
            "enable_interim_results": self.enable_interim_results,
            "enable_speaker_diarization": self._enable_speaker_diarization,
            "diarization_speaker_count": self._diarization_speaker_count,
            "single_utterance": self._single_utterance,
            "speech_contexts": self._speech_contexts,
        }