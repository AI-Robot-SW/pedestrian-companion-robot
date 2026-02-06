"""
Minimal local stand-in for om1_speech.

Provides AudioInputStream using PyAudio so existing providers/examples can run
without the external om1_speech package.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional, Tuple
import time

import torch
import pyaudio


class AudioInputStream:
    """
    Microphone audio input stream wrapper.

    Notes
    -----
    - If remote_input=True, audio data should be pushed via fill_buffer_remote().
    - device_name is best-effort matched to a PyAudio input device when device is None.
    """

    def __init__(
        self,
        rate: Optional[int] = None,
        chunk: Optional[int] = None,
        device: Optional[int] = None,
        device_name: Optional[str] = None,
        audio_data_callback: Optional[
            Callable[[bytes, int, dict, int], None]
        ] = None,
        language_code: Optional[str] = None,
        remote_input: bool = False,
        enable_tts_interrupt: bool = False,
    ) -> None:
        self.rate = rate or 16000
        self.chunk = chunk or 1024
        self.device = device
        self.device_name = device_name
        self.language_code = language_code
        self.remote_input = remote_input
        self.enable_tts_interrupt = enable_tts_interrupt

        self.running = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[bytes, int, dict, int], None]] = []
        if audio_data_callback is not None:
            self._callbacks.append(audio_data_callback)

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    def register_audio_data_callback(
        self, callback: Callable[[bytes, int, dict, int], None]
    ) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def _resolve_device_index(self) -> Optional[int]:
        if self.device is not None:
            return self.device
        if not self.device_name:
            return None
        if self._pa is None:
            return None
        name_l = self.device_name.lower()
        for idx in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(idx)
            dev_name = str(info.get("name", "")).lower()
            if name_l in dev_name and int(info.get("maxInputChannels", 0)) > 0:
                return idx
        logging.warning(f"AudioInputStream device_name not found: {self.device_name}")
        return None

    def _on_audio_data(self, in_data, frame_count, time_info, status_flags):
        for cb in list(self._callbacks):
            try:
                cb(in_data, frame_count, time_info, status_flags)
            except Exception as exc:
                logging.error(f"AudioInputStream callback error: {exc}")
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        with self._lock:
            if self.running:
                logging.warning("AudioInputStream already running")
                return
            self.running = True

        if self.remote_input:
            logging.info("AudioInputStream started in remote_input mode")
            return

        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        device_index = self._resolve_device_index()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=device_index,
            stream_callback=self._on_audio_data,
        )
        logging.info(
            "AudioInputStream started: rate=%s chunk=%s device=%s",
            self.rate,
            self.chunk,
            device_index if device_index is not None else "default",
        )

    def stop(self) -> None:
        with self._lock:
            if not self.running:
                logging.warning("AudioInputStream not running")
                return
            self.running = False

        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        logging.info("AudioInputStream stopped")

    def fill_buffer_remote(self, audio_bytes: bytes) -> None:
        """
        Feed audio bytes when remote_input is enabled.
        """
        if not self.remote_input:
            return
        for cb in list(self._callbacks):
            try:
                cb(audio_bytes, len(audio_bytes), {}, 0)
            except Exception as exc:
                logging.error(f"AudioInputStream callback error: {exc}")


class SileroVAD:
    """
    Streaming Silero VAD wrapper using VADIterator.

    Usage
    -----
    vad = SileroVAD(sampling_rate=16000, threshold=0.5, chunk_size=1024)
    event = vad.process(audio_chunk)
    if event:
        ...
    if vad.speech_active:
        ...
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        min_speech_duration_ms: int = 250,
        chunk_size: Optional[int] = None,
        device: str = "cpu",
        return_seconds: bool = False,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        if self.sampling_rate == 16000:
            required_samples = 512
        elif self.sampling_rate == 8000:
            required_samples = 256
        else:
            raise ValueError(
                "SileroVAD supports sampling_rate 8000 or 16000 only."
            )
        if chunk_size not in (None, required_samples):
            logging.warning(
                "SileroVAD chunk_size=%s does not match required %s for %s Hz. "
                "Overriding to %s.",
                chunk_size,
                required_samples,
                self.sampling_rate,
                required_samples,
            )
        self.chunk_size = required_samples
        self.device = device
        self.return_seconds = return_seconds

        self._speech_active = False
        self._buffer = bytearray()
        self._prob_buffer = bytearray()
        self._prob_model = None
        self._last_prob: Optional[float] = None

        try:
            from silero_vad import VADIterator, load_silero_vad
        except Exception as exc:
            raise RuntimeError(
                "silero_vad package is required for SileroVAD. "
                "Install silero-vad and its dependencies."
            ) from exc

        try:
            self._model = load_silero_vad(device=device)
        except TypeError:
            self._model = load_silero_vad()
            if hasattr(self._model, "to"):
                self._model.to(device)

        try:
            self._vad_iterator = VADIterator(
                self._model,
                sampling_rate=self.sampling_rate,
                threshold=self.threshold,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                min_speech_duration_ms=self.min_speech_duration_ms,
            )
        except TypeError:
            self._vad_iterator = VADIterator(
                self._model,
                sampling_rate=self.sampling_rate,
                threshold=self.threshold,
            )

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def last_prob(self) -> Optional[float]:
        return self._last_prob

    def reset(self) -> None:
        if hasattr(self._vad_iterator, "reset_states"):
            self._vad_iterator.reset_states()
        elif hasattr(self._vad_iterator, "reset"):
            self._vad_iterator.reset()
        if self._prob_model is not None and hasattr(self._prob_model, "reset_states"):
            self._prob_model.reset_states()
        self._speech_active = False
        self._buffer.clear()
        self._prob_buffer.clear()
        self._last_prob = None

    def _iter_chunks(self, audio_chunk: bytes, buffer: Optional[bytearray] = None):
        target_buffer = self._buffer if buffer is None else buffer
        if self.chunk_size is None:
            yield audio_chunk
            return

        frame_bytes = self.chunk_size * 2
        target_buffer.extend(audio_chunk)
        while len(target_buffer) >= frame_bytes:
            chunk = bytes(target_buffer[:frame_bytes])
            del target_buffer[:frame_bytes]
            yield chunk

    def _bytes_to_tensor(self, audio_chunk: bytes):
        import numpy as np
        import torch

        if len(audio_chunk) % 2 == 1:
            audio_chunk = audio_chunk[:-1]
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        return tensor

    def process(self, audio_chunk: bytes):
        if not audio_chunk:
            return None
        import torch

        last_event = None
        for chunk in self._iter_chunks(audio_chunk):
            if not chunk:
                continue
            audio_tensor = self._bytes_to_tensor(chunk)
            with torch.no_grad():
                event = self._vad_iterator(audio_tensor, return_seconds=self.return_seconds)
            if event:
                if "start" in event:
                    self._speech_active = True
                if "end" in event:
                    self._speech_active = False
                last_event = event
        return last_event

    def _get_prob_model(self):
        if self._prob_model is not None:
            return self._prob_model
        try:
            from silero_vad import load_silero_vad
        except Exception as exc:
            raise RuntimeError(
                "silero_vad package is required for SileroVAD. "
                "Install silero-vad and its dependencies."
            ) from exc

        try:
            self._prob_model = load_silero_vad(device=self.device)
        except TypeError:
            self._prob_model = load_silero_vad()
            if hasattr(self._prob_model, "to"):
                self._prob_model.to(self.device)
        return self._prob_model

    def process_prob(self, audio_chunk: bytes) -> Optional[float]:
        """
        Return speech probability for the latest processed chunk.

        Note: Uses a separate model instance to avoid interfering with VADIterator state.
        """
        if not audio_chunk:
            return None

        import torch

        model = self._get_prob_model()
        last_prob = None
        for chunk in self._iter_chunks(audio_chunk, buffer=self._prob_buffer):
            if not chunk:
                continue
            audio_tensor = self._bytes_to_tensor(chunk)
            with torch.no_grad():
                last_prob = model(audio_tensor, self.sampling_rate).item()

        if last_prob is not None:
            self._last_prob = last_prob
        return last_prob


class LatencyTracker:
    def __init__(self, sample_rate: int, channels: int = 1, sample_width_bytes: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.stream_start_ts = None
        self.samples_sent = 0
        self.last_audio_end_ts = None

    def on_start(self):
        self.stream_start_ts = time.perf_counter()
        self.samples_sent = 0
        self.last_audio_end_ts = self.stream_start_ts

    def on_send_audio(self, audio_chunk: bytes):
        if self.stream_start_ts is None:
            self.on_start()
        samples = len(audio_chunk) // (self.sample_width_bytes * self.channels)
        self.samples_sent += samples
        # 오디오 타임라인 기준 “마지막 오디오가 끝나는 시각”
        self.last_audio_end_ts = self.stream_start_ts + (self.samples_sent / self.sample_rate)

    def on_result(self):
        if self.last_audio_end_ts is None:
            return None
        now = time.perf_counter()
        latency = now - self.last_audio_end_ts
        return max(0.0, latency)
