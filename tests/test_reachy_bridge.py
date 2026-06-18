"""Unit tests for ``reachy_bridge`` helpers and playback completion logic."""

from __future__ import annotations

import base64
import io
import json
import socket
import threading
import wave

import numpy as np
import pytest

from reachy_bridge import (
    ReachyBridge,
    _chunk_rms,
    _wait_tcp_port,
    decode_wav_to_samples,
)


def _pcm16_wav_b64(*, sample_rate: int, samples: np.ndarray) -> str:
    """Build a mono int16 WAV, return base64 as the server sends.

    Args:
        sample_rate: WAV sample rate in Hz.
        samples: Mono float32 or float64 samples roughly in [-1, 1].

    Returns:
        Base64-encoded WAV bytes.
    """
    buf = io.BytesIO()
    pcm = np.clip(samples, -1.0, 1.0)
    frames = (pcm * 32767.0).astype(np.int16).tobytes()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_chunk_rms_empty() -> None:
    """Empty chunk has zero RMS."""
    assert _chunk_rms(np.array([], dtype=np.float32)) == 0.0


def test_chunk_rms_constant() -> None:
    """Constant amplitude chunk matches closed-form RMS."""
    chunk = np.full(100, 0.5, dtype=np.float32)
    assert abs(_chunk_rms(chunk) - 0.5) < 1e-6


def test_decode_wav_int16_mono() -> None:
    """Decoded int16 WAV is mono float32 near [-1, 1]."""
    sr = 16000
    t = np.linspace(0, 0.01, 160, endpoint=False, dtype=np.float64)
    samples = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    b64 = _pcm16_wav_b64(sample_rate=sr, samples=samples)
    audio, out_sr = decode_wav_to_samples(b64)
    assert out_sr == sr
    assert audio.shape == samples.shape
    assert float(np.max(np.abs(audio))) <= 1.0
    assert float(np.max(np.abs(audio))) > 0.01


@pytest.mark.asyncio
async def test_try_send_frontend_playback_complete_sends_once() -> None:
    """Completion message is sent once when synth finished and no pending audio."""

    class _Ws:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

    bridge = ReachyBridge("ws://example.invalid", reachy=None)
    bridge._saw_backend_synth_complete = True
    bridge._pending_audio_playbacks = 0
    ws = _Ws()
    await bridge._try_send_frontend_playback_complete(ws)
    await bridge._try_send_frontend_playback_complete(ws)
    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "frontend-playback-complete"


@pytest.mark.asyncio
async def test_try_send_frontend_playback_complete_blocked_by_pending() -> None:
    """No message while audio chunks are still marked pending."""

    class _Ws:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

    bridge = ReachyBridge("ws://example.invalid", reachy=None)
    bridge._saw_backend_synth_complete = True
    bridge._pending_audio_playbacks = 1
    ws = _Ws()
    await bridge._try_send_frontend_playback_complete(ws)
    assert ws.sent == []


@pytest.mark.asyncio
async def test_try_send_frontend_playback_complete_without_synth() -> None:
    """No message if ``backend-synth-complete`` was never observed."""

    class _Ws:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

    bridge = ReachyBridge("ws://example.invalid", reachy=None)
    bridge._saw_backend_synth_complete = False
    bridge._pending_audio_playbacks = 0
    ws = _Ws()
    await bridge._try_send_frontend_playback_complete(ws)
    assert ws.sent == []


def test_wait_tcp_port_opens() -> None:
    """Port probe succeeds once a listener accepts TCP."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    host, port = sock.getsockname()[:2]

    def _serve() -> None:
        conn, _ = sock.accept()
        conn.close()

    threading.Thread(target=_serve, daemon=True).start()
    try:
        ok = _wait_tcp_port(host, int(port), total_timeout=5.0)
        assert ok is True
    finally:
        sock.close()


@pytest.mark.asyncio
async def test_playback_complete_after_straggle_sends() -> None:
    """Straggler coroutine emits completion after its delay when state allows."""

    class _Ws:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

    bridge = ReachyBridge("ws://example.invalid", reachy=None)
    bridge._saw_backend_synth_complete = True
    bridge._pending_audio_playbacks = 0
    ws = _Ws()
    await bridge._playback_complete_after_straggle(ws)
    assert any(m.get("type") == "frontend-playback-complete" for m in ws.sent)
