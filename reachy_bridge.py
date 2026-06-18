"""
Reachy Mini ↔ Open-LLM-VTuber Bridge Client
=============================================
WebSocket client that connects to Open-LLM-VTuber's /client-ws endpoint,
receives audio + emotion expressions, and drives a Reachy Mini Lite robot.

Usage:
    uv run reachy_bridge.py                  # MuJoCo simulation (spawns local daemon)
    uv run reachy_bridge.py --usb            # Connect to an already-running robot daemon
    uv run reachy_bridge.py --text-mode      # Keyboard text input (no mic)
    uv run reachy_bridge.py --server ws://192.168.1.100:12393/client-ws

Optional: route emotions through your own mapping::

    export OLV_REACHY_EMOTION_CALLABLE=my_package.chat:emotion_to_action

The callable must accept ``(reachy, emotion: str)`` where *emotion* is one of:
``neutral``, ``sadness``, ``anger``, ``joy``.

Requires: ``uv sync --extra reachy`` (``reachy-mini[mujoco]``, ``websockets``,
``sounddevice``, plus core deps such as ``numpy``). See ``REACHY.md`` for details.

Voice: the bridge sends ``mic-audio-data`` while you speak and ``mic-audio-end``
after ~1.5s of silence so the server can run ASR (same contract as the web UI).
After TTS, the server waits for ``frontend-playback-complete``; the bridge sends
that automatically when local playback finishes (otherwise the server would hang).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import importlib
import io
import json
import os
import signal
import socket
import sys
import time
import wave
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from loguru import logger

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment, misc]

try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

# ---------------------------------------------------------------------------
# Emotion → Reachy action mapping
# ---------------------------------------------------------------------------
# Open-LLM-VTuber sends expression INDEX integers (from model_dict.json's
# emotionMap). The default mao_pro mapping is:
#
#   neutral → 0,  fear/sadness → 1,  anger/disgust → 2,  joy/smirk/surprise → 3

SAMPLE_RATE = 16000  # Open-LLM-VTuber expects 16 kHz

# Mic utterance detection: server only runs ASR after ``mic-audio-end`` (see
# ``websocket_handler._handle_audio_data`` / ``handle_conversation_trigger``).
_MIC_CHUNK_S = 0.1
_MIC_MIN_SPEECH_CHUNKS = 3  # ~0.3 s above threshold before we start buffering
_MIC_SILENCE_END_CHUNKS = 15  # ~1.5 s quiet after speech → submit utterance
_MIC_SPEECH_RMS_THRESHOLD = 0.012  # fallback if env / calibration skipped


class _EmotionHookCache:
    """Stores the lazily imported user emotion hook and resolution flag."""

    __slots__ = ("hook", "resolved")

    def __init__(self) -> None:
        self.resolved = False
        self.hook: Callable[[Any, str], None] | None = None


_EMOTION_HOOK_CACHE = _EmotionHookCache()


def _resolve_emotion_hook() -> Callable[[Any, str], None] | None:
    """Load ``OLV_REACHY_EMOTION_CALLABLE=module.path:function_name`` once.

    Returns:
        A ``(reachy, emotion) -> None`` callable, or None if unset or invalid.
    """
    if _EMOTION_HOOK_CACHE.resolved:
        return _EMOTION_HOOK_CACHE.hook
    _EMOTION_HOOK_CACHE.resolved = True
    spec = os.environ.get("OLV_REACHY_EMOTION_CALLABLE", "").strip()
    if not spec:
        return None
    if ":" not in spec:
        logger.error(
            "OLV_REACHY_EMOTION_CALLABLE must look like 'my.module:func_name', got {!r}",
            spec,
        )
        return None
    mod_path, _, attr = spec.partition(":")
    try:
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, attr)
    except (ImportError, AttributeError):
        logger.exception("Failed to import emotion hook {!r}", spec)
        return None
    if not callable(fn):
        logger.error("OLV_REACHY_EMOTION_CALLABLE target {!r} is not callable", spec)
        return None
    _EMOTION_HOOK_CACHE.hook = cast(Callable[[Any, str], None], fn)
    logger.info("Using custom Reachy emotion hook: {}", spec)
    return _EMOTION_HOOK_CACHE.hook


def _builtin_emotion_pose(reachy: Any, emotion: str) -> None:
    """Drive the head with small safe poses via the official SDK (if installed).

    Args:
        reachy: Connected ``ReachyMini`` instance.
        emotion: One of ``neutral``, ``sadness``, ``anger``, ``joy``.
    """
    try:
        from reachy_mini.utils import create_head_pose
    except ImportError:
        logger.debug(
            "reachy_mini.utils not importable; skipping builtin pose ({})", emotion
        )
        return
    if not hasattr(reachy, "goto_target"):
        return

    presets: dict[str, tuple[float, float, float, float, float, float, float]] = {
        # x, y, z_mm, roll_deg, pitch_deg, yaw_deg, duration_s
        "neutral": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35),
        "sadness": (0.0, 0.0, -8.0, 0.0, -14.0, 0.0, 0.55),
        "anger": (0.0, 0.0, 6.0, 0.0, 12.0, 0.0, 0.28),
        "joy": (0.0, 0.0, 12.0, 10.0, -8.0, 8.0, 0.42),
    }
    row = presets.get(emotion)
    if row is None:
        return
    x, y, z, roll, pitch, yaw, duration = row
    head = create_head_pose(
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        mm=True,
        degrees=True,
    )
    reachy.goto_target(head=head, duration=duration)


def apply_emotion_to_reachy(reachy: Any, emotion: str) -> None:
    """Apply one emotion label to the robot (user hook or builtin).

    Args:
        reachy: Live ``ReachyMini`` instance (or any object your hook expects).
        emotion: Canonical label ``neutral`` / ``sadness`` / ``anger`` / ``joy``.
    """
    hook = _resolve_emotion_hook()
    if hook is not None:
        try:
            hook(reachy, emotion)
        except Exception:
            logger.exception("Custom emotion hook failed for {}", emotion)
        return
    _builtin_emotion_pose(reachy, emotion)


def reachy_neutral(reachy: Any) -> None:
    """Return to a neutral head pose."""
    apply_emotion_to_reachy(reachy, "neutral")


def reachy_sadness(reachy: Any) -> None:
    """Express low-energy / negative valence (fear, sadness indices)."""
    apply_emotion_to_reachy(reachy, "sadness")


def reachy_anger(reachy: Any) -> None:
    """Express high-arousal negative (anger, disgust indices)."""
    apply_emotion_to_reachy(reachy, "anger")


def reachy_joy(reachy: Any) -> None:
    """Express positive valence (joy, smirk, surprise indices)."""
    apply_emotion_to_reachy(reachy, "joy")


# Maps expression index → action function (mao_pro model_dict.json indices)
EXPRESSION_ACTIONS: dict[int, Callable[[Any], None]] = {
    0: reachy_neutral,
    1: reachy_sadness,
    2: reachy_anger,
    3: reachy_joy,
}

# Maps emotion name strings → action functions (for messages that send names)
EMOTION_NAME_ACTIONS: dict[str, Callable[[Any], None]] = {
    "neutral": reachy_neutral,
    "fear": reachy_sadness,
    "sadness": reachy_sadness,
    "anger": reachy_anger,
    "disgust": reachy_anger,
    "joy": reachy_joy,
    "smirk": reachy_joy,
    "surprise": reachy_joy,
}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def decode_wav_to_samples(b64_audio: str) -> tuple[np.ndarray, int]:
    """Decode base64 WAV audio from the server into float32 numpy array.

    Args:
        b64_audio: Base64-encoded WAV bytes from the server.

    Returns:
        A pair ``(samples, sample_rate)`` with mono float32 samples in [-1, 1].
    """
    raw = base64.b64decode(b64_audio)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    if sampwidth == 2:
        audio /= 32768.0
    elif sampwidth == 4:
        # 32-bit PCM from pydub/ffmpeg is not full int32 range; scale conservatively
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 1.5:
            audio = np.clip(audio, -1.0, 1.0)
        else:
            audio /= float(np.iinfo(np.int32).max)
    return audio, sr


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    """Play audio through the default output device.

    Args:
        audio: Mono float32 samples.
        sample_rate: Playback sample rate in Hz.
    """
    if not HAS_SOUNDDEVICE:
        print("[WARN] sounddevice not installed, skipping audio playback")
        return
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def _chunk_rms(chunk: np.ndarray) -> float:
    """Root-mean-square level of a mono float32 chunk.

    Args:
        chunk: Audio samples.

    Returns:
        RMS amplitude in the same scale as the samples.
    """
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))


def record_chunk(duration_s: float = _MIC_CHUNK_S) -> np.ndarray:
    """Record a short audio chunk from the default mic.

    Args:
        duration_s: Length of the capture window in seconds.

    Returns:
        Float32 mono samples at ``SAMPLE_RATE``, or empty array if no mic backend.
    """
    if not HAS_SOUNDDEVICE:
        return np.array([], dtype=np.float32)
    frames = int(SAMPLE_RATE * duration_s)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


# ---------------------------------------------------------------------------
# Reachy connection
# ---------------------------------------------------------------------------

_REACHY_SDK_CONNECT_TIMEOUT_S = 60.0
_REACHY_DAEMON_PORT_WAIT_S = 90.0


def _wait_tcp_port(host: str, port: int, *, total_timeout: float) -> bool:
    """Block until ``host:port`` accepts TCP or ``total_timeout`` elapses.

    Args:
        host: Hostname or IP to probe.
        port: TCP port.
        total_timeout: Maximum seconds to keep trying.

    Returns:
        True if a connection succeeded at least once, False on timeout.
    """
    deadline = time.monotonic() + total_timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _terminate_reachy_sim_daemons() -> None:
    """Send SIGTERM to local ``reachy-mini-daemon`` processes running with ``--sim``.

    Used when the bridge exits or gives up connecting, so a fire-and-forget
    ``Popen`` from the SDK does not leave port 8000 occupied.
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed; cannot auto-stop reachy-mini-daemon.")
        return
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
            flat = " ".join(str(c) for c in cmd)
            if "reachy-mini-daemon" not in flat or "--sim" not in flat:
                continue
            logger.info("Stopping Reachy sim daemon pid={}", proc.pid)
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            logger.exception("Failed to signal reachy-mini-daemon pid={}", proc.pid)
    time.sleep(0.5)
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
            flat = " ".join(str(c) for c in cmd)
            if "reachy-mini-daemon" not in flat or "--sim" not in flat:
                continue
            logger.warning(
                "Reachy sim daemon still running pid={}, sending SIGKILL", proc.pid
            )
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            logger.exception("Failed to kill reachy-mini-daemon pid={}", proc.pid)


def terminate_reachy_sim_daemons_bounded(timeout_s: float = 6.0) -> None:
    """Like :func:`_terminate_reachy_sim_daemons` but does not block process exit forever."""
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        fut = ex.submit(_terminate_reachy_sim_daemons)
        try:
            fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "reachy-mini-daemon cleanup exceeded {:.1f}s; continuing exit.",
                timeout_s,
            )


def connect_reachy(
    *,
    use_usb: bool = False,
    reachy_host: str | None = None,
    reachy_daemon_port: int = 8000,
) -> tuple[Any | None, bool]:
    """Connect to the Reachy Mini (simulated daemon or existing hardware daemon).

    Args:
        use_usb: When False, spawn a local MuJoCo simulation daemon. When True,
            connect to a daemon that is already running (typically with the
            physical robot or a manually started simulator).
        reachy_host: Optional daemon hostname or IP (implies remote connection).
        reachy_daemon_port: HTTP port of the Reachy Mini daemon (default 8000).

    Returns:
        Tuple of ``(ReachyMini instance or None, manage_sim_daemon)``.
        ``manage_sim_daemon`` is True when this bridge started or expects local
        MuJoCo sim; callers may stop ``reachy-mini-daemon --sim`` on exit.
    """
    try:
        from reachy_mini import ReachyMini
    except ImportError:
        logger.warning(
            "reachy-mini SDK not installed — emotions will only log unless a hook runs "
            "without the SDK."
        )
        return None, False

    manage_sim = not use_usb

    try:
        if use_usb:
            if reachy_host:
                robot = ReachyMini(
                    spawn_daemon=False,
                    use_sim=False,
                    host=reachy_host,
                    port=reachy_daemon_port,
                    connection_mode="network",
                )
            else:
                robot = ReachyMini(
                    spawn_daemon=False,
                    use_sim=False,
                    port=reachy_daemon_port,
                    connection_mode="auto",
                )
            logger.info(
                "ReachyMini connected to hardware daemon (USB / existing daemon)."
            )
            return robot, False

        # Local MuJoCo: the SDK spawns ``reachy-mini-daemon`` via Popen and then
        # connects immediately — the daemon often is not listening yet, which
        # produced ConnectionError and left an orphan daemon (see debug.log).
        sim_kwargs = {
            "spawn_daemon": True,
            "use_sim": True,
            "port": reachy_daemon_port,
            "connection_mode": "localhost_only",
            "timeout": _REACHY_SDK_CONNECT_TIMEOUT_S,
            "media_backend": "no_media",
        }
        try:
            robot = ReachyMini(**sim_kwargs)
        except Exception as first_err:
            logger.warning(
                "ReachyMini first connect failed (daemon may still be starting): {}",
                first_err,
            )
            if not _wait_tcp_port(
                "127.0.0.1",
                reachy_daemon_port,
                total_timeout=_REACHY_DAEMON_PORT_WAIT_S,
            ):
                logger.error(
                    "Timed out waiting for Reachy daemon on 127.0.0.1:{}",
                    reachy_daemon_port,
                )
                _terminate_reachy_sim_daemons()
                return None, manage_sim
            try:
                robot = ReachyMini(**sim_kwargs)
            except Exception as e2:
                logger.exception("Failed to connect to Reachy after wait: {}", e2)
                _terminate_reachy_sim_daemons()
                return None, manage_sim

        logger.info("ReachyMini connected (MuJoCo simulation daemon).")
        return robot, manage_sim
    except Exception as e:
        logger.exception("Failed to connect to Reachy: {}", e)
        if manage_sim:
            _terminate_reachy_sim_daemons()
        return None, manage_sim


def stop_sounddevice() -> None:
    """Abort any in-flight ``sounddevice`` capture/playback (non-fatal if unused)."""
    if not HAS_SOUNDDEVICE:
        return
    try:
        sd.stop()
    except Exception:
        pass


def shutdown_reachy(reachy: Any) -> None:
    """Release Reachy client resources without blocking on a dead daemon.

    Disconnects the SDK WebSocket first. A full ``__exit__`` (which closes
    ``media_manager`` and can block when the sim process is already gone) runs
    only if ``OLV_REACHY_FULL_EXIT=1`` is set.

    Args:
        reachy: Instance returned by ``connect_reachy``, or None.
    """
    if reachy is None:
        return
    try:
        client = getattr(reachy, "client", None)
        disconnect = getattr(client, "disconnect", None) if client is not None else None
        if callable(disconnect):
            disconnect()
    except Exception:
        logger.exception("Reachy client.disconnect failed during shutdown")
    if os.environ.get("OLV_REACHY_FULL_EXIT", "").strip() == "1":
        try:
            exit_fn = getattr(reachy, "__exit__", None)
            if callable(exit_fn):
                exit_fn(None, None, None)
        except Exception:
            logger.exception("Reachy __exit__ failed during shutdown")


def prime_reachy_for_bridge(reachy: Any) -> None:
    """Enable motor torque so head poses and motion commands take effect.

    Args:
        reachy: Connected ``ReachyMini`` instance, or None (no-op).
    """
    if reachy is None:
        return
    enable = getattr(reachy, "enable_motors", None)
    if not callable(enable):
        logger.warning(
            "ReachyMini.enable_motors not found; robot motion may stay idle."
        )
        return
    try:
        enable()
        logger.info("Reachy motors enabled (torque on).")
    except Exception:
        logger.exception("Reachy enable_motors() failed.")


# ---------------------------------------------------------------------------
# Bridge client
# ---------------------------------------------------------------------------


class ReachyBridge:
    """WebSocket bridge between Open-LLM-VTuber and Reachy Mini Lite."""

    def __init__(self, server_url: str, reachy: Any, text_mode: bool = False):
        self.server_url = server_url
        self.reachy = reachy
        self.text_mode = text_mode
        self._playing = asyncio.Event()
        self._playing.set()  # start as "not playing"
        self._ws: Any | None = None
        self._saw_backend_synth_complete = False
        self._pending_audio_playbacks = 0
        self._sent_playback_complete_this_chain = False
        self._playback_straggler_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._exit_via_signal = False

    def _cancel_playback_straggler(self) -> None:
        if self._playback_straggler_task and not self._playback_straggler_task.done():
            self._playback_straggler_task.cancel()
        self._playback_straggler_task = None

    async def _send_json(self, ws: Any, payload: dict) -> None:
        """Send JSON to the server with optional debug logging."""
        if os.environ.get("OLV_BRIDGE_DEBUG_WS", "").strip() == "1":
            logger.info("[BRIDGE] → server type={}", payload.get("type"))
        await ws.send(json.dumps(payload))

    async def run(self) -> None:
        """Main loop: connect, listen, and drive the robot."""
        if websockets is None:
            raise RuntimeError(
                "Missing dependency: install websockets (e.g. uv sync --extra reachy)."
            )
        self._shutdown.clear()
        self._exit_via_signal = False
        loop = asyncio.get_running_loop()
        if os.name == "posix":
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, self._shutdown.set)
                except (NotImplementedError, AttributeError, ValueError):
                    break

        print(f"[BRIDGE] Connecting to {self.server_url} ...")
        async with websockets.connect(self.server_url) as ws:
            self._ws = ws
            print("[BRIDGE] Connected!")
            recv_task = asyncio.create_task(self._receive_loop(ws))
            side_task = asyncio.create_task(
                self._text_input_loop(ws)
                if self.text_mode
                else self._mic_input_loop(ws)
            )
            stop_task = asyncio.create_task(self._shutdown.wait())
            try:
                done, _ = await asyncio.wait(
                    {recv_task, side_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done:
                    self._exit_via_signal = True
                    print("\n[BRIDGE] Interrupt: closing WebSocket and stopping audio…")
            finally:
                stop_sounddevice()
                for t in (recv_task, side_task, stop_task):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(
                    recv_task, side_task, stop_task, return_exceptions=True
                )
                self._cancel_playback_straggler()
                try:
                    await ws.close()
                except Exception:
                    pass
                self._ws = None

        if os.name == "posix":
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, AttributeError, ValueError):
                    break

    # -- Receiving from server -----------------------------------------------

    async def _receive_loop(self, ws: Any) -> None:
        """Listen for messages from Open-LLM-VTuber."""
        async for raw in ws:
            try:
                text = raw if isinstance(raw, str) else raw.decode("utf-8")
                msg = json.loads(text)
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
                logger.warning(
                    "Skipping non-JSON WebSocket frame: {} ({})", e, type(raw)
                )
                continue
            msg_type = msg.get("type")
            if os.environ.get("OLV_BRIDGE_DEBUG_WS", "").strip() == "1":
                logger.info("[BRIDGE] ← server type={}", msg_type)
            if msg_type == "audio":
                await self._handle_audio(ws, msg)
            elif msg_type == "control":
                self._handle_control(msg)
            elif msg_type == "full-text":
                print(f"[AI] {msg.get('text', '')}")
            elif msg_type == "user-input-transcription":
                print(f"[YOU] {msg.get('text', '')}")
            elif msg_type == "set-model-and-conf":
                model_info = msg.get("model_info", {})
                print(f"[BRIDGE] Model: {model_info.get('name', '?')}")
            elif msg_type == "error":
                print(f"[ERROR] {msg.get('message', 'unknown')}")
            elif msg_type == "heartbeat-ack":
                pass
            elif msg_type == "tool_call_status":
                tool = msg.get("tool_name", "?")
                status = msg.get("status", "?")
                print(f"[TOOL] {tool}: {status}")
            elif msg_type == "backend-synth-complete":
                self._saw_backend_synth_complete = True
                if (
                    self._playback_straggler_task
                    and not self._playback_straggler_task.done()
                ):
                    self._playback_straggler_task.cancel()
                self._playback_straggler_task = asyncio.create_task(
                    self._playback_complete_after_straggle(ws)
                )
            else:
                logger.debug("Unhandled server message type: {}", msg_type)

    def _reset_playback_completion_state(self) -> None:
        """Prepare for a new AI conversation chain (server is about to stream TTS)."""
        self._cancel_playback_straggler()
        self._saw_backend_synth_complete = False
        self._sent_playback_complete_this_chain = False

    async def _playback_complete_after_straggle(self, ws: Any) -> None:
        """Wait briefly so late ``audio`` frames after ``backend-synth-complete`` still play."""
        try:
            await asyncio.sleep(0.35)
            await self._try_send_frontend_playback_complete(ws)
        except asyncio.CancelledError:
            return

    async def _try_send_frontend_playback_complete(self, ws: Any) -> None:
        """Tell the server local playback finished so the conversation can finish.

        ``finalize_conversation_turn`` waits indefinitely for this message after
        ``backend-synth-complete``; the web UI sends it when audio finishes.
        """
        if self._sent_playback_complete_this_chain:
            return
        if not self._saw_backend_synth_complete:
            return
        if self._pending_audio_playbacks > 0:
            return
        self._sent_playback_complete_this_chain = True
        await self._send_json(ws, {"type": "frontend-playback-complete"})
        logger.info("Sent frontend-playback-complete (server conversation can finish).")

    async def _handle_audio(self, ws: Any, msg: dict) -> None:
        """Decode audio, play it, and trigger emotion actions."""
        b64_audio = msg.get("audio")
        actions = msg.get("actions") or {}
        expressions = actions.get("expressions") or []
        display_text = msg.get("display_text") or {}
        text = display_text.get("text", "")

        if text:
            print(f"[AI] {text}")

        if expressions:
            self._apply_expressions(expressions)

        if not b64_audio:
            await self._try_send_frontend_playback_complete(ws)
            return

        self._pending_audio_playbacks += 1
        self._playing.clear()
        try:
            audio, sr = decode_wav_to_samples(b64_audio)
            await asyncio.get_event_loop().run_in_executor(None, play_audio, audio, sr)
        except Exception as e:
            print(f"[ERROR] Audio playback failed: {e}")
        finally:
            self._playing.set()
            self._pending_audio_playbacks -= 1
            await self._try_send_frontend_playback_complete(ws)

    def _apply_expressions(self, expressions: list) -> None:
        """Translate expression indices/names to Reachy actions."""
        for expr in expressions:
            if isinstance(expr, int):
                action_fn = EXPRESSION_ACTIONS.get(expr)
                label = f"index={expr}"
            elif isinstance(expr, str):
                action_fn = EMOTION_NAME_ACTIONS.get(expr.lower())
                label = expr
            else:
                continue

            if action_fn and self.reachy:
                try:
                    action_fn(self.reachy)
                except Exception as e:
                    print(f"[ERROR] Reachy action failed ({label}): {e}")
            else:
                print(f"[EMOTION] {label}")

    def _handle_control(self, msg: dict) -> None:
        """Handle control messages from server."""
        command = msg.get("text", "")
        if command == "conversation-chain-start":
            self._reset_playback_completion_state()
            print("[BRIDGE] AI is thinking...")
        elif command == "conversation-chain-end":
            print("[BRIDGE] AI response complete.")
        elif command == "interrupt":
            print("[BRIDGE] Interruption detected.")

    async def _calibrate_mic_rms_threshold(self) -> float:
        """Measure ambient noise and derive an RMS threshold above the noise floor.

        Returns:
            RMS threshold for ``loud = rms >= thr``. Skipped when
            ``OLV_BRIDGE_SPEECH_RMS`` is set in the environment.
        """
        env = os.environ.get("OLV_BRIDGE_SPEECH_RMS", "").strip()
        if env:
            return float(env)
        rms_samples: list[float] = []
        for _ in range(18):
            chunk = await asyncio.get_event_loop().run_in_executor(
                None, record_chunk, _MIC_CHUNK_S
            )
            rms_samples.append(_chunk_rms(chunk))
        arr = np.asarray(rms_samples, dtype=np.float64)
        # Median is inflated if the user talks during calibration; 25th percentile
        # tracks quiet noise better while ignoring brief spikes.
        floor = float(np.percentile(arr, 25)) if arr.size else 0.001
        thr = max(0.004, min(0.12, floor * 4.5))
        print(
            "[BRIDGE] Mic calibrated (~1.8s): ambient noise floor (p25 RMS)≈"
            f"{floor:.4f}, speech threshold≈{thr:.4f} "
            "(quiet room; set OLV_BRIDGE_SPEECH_RMS to override)"
        )
        return thr

    # -- Microphone input (auto mode) ----------------------------------------

    async def _mic_input_loop(self, ws: Any) -> None:
        """Record mic, stream ``mic-audio-data`` during speech, then ``mic-audio-end``.

        The server only flushes buffered audio to ASR when it receives
        ``mic-audio-end``; streaming data alone never starts a conversation.
        """
        thr = await self._calibrate_mic_rms_threshold()
        silence_end = int(
            os.environ.get("OLV_BRIDGE_SILENCE_CHUNKS", str(_MIC_SILENCE_END_CHUNKS))
        )
        min_speech = int(
            os.environ.get("OLV_BRIDGE_MIN_SPEECH_CHUNKS", str(_MIC_MIN_SPEECH_CHUNKS))
        )
        print(
            "[BRIDGE] Speak toward the mic, then pause ~1.5s — you should see "
            f"[YOU] then [AI] (silence chunks={silence_end})."
        )
        in_utterance = False
        quiet_run = 0
        pending_loud: list[np.ndarray] = []
        dbg_tick = 0

        while True:
            await self._playing.wait()

            chunk = await asyncio.get_event_loop().run_in_executor(
                None, record_chunk, _MIC_CHUNK_S
            )
            if chunk.size == 0:
                await asyncio.sleep(_MIC_CHUNK_S)
                continue

            rms = _chunk_rms(chunk)
            loud = rms >= thr
            dbg_tick += 1
            if (
                os.environ.get("OLV_BRIDGE_DEBUG_MIC", "").strip() == "1"
                and dbg_tick % 25 == 0
            ):
                print(
                    f"[BRIDGE] mic RMS={rms:.5f} thr={thr:.5f} "
                    f"in_utt={in_utterance} loud={loud}"
                )

            if not in_utterance:
                if loud:
                    pending_loud.append(chunk)
                    if len(pending_loud) >= min_speech:
                        in_utterance = True
                        quiet_run = 0
                        for p in pending_loud:
                            await self._send_json(
                                ws, {"type": "mic-audio-data", "audio": p.tolist()}
                            )
                        pending_loud.clear()
                else:
                    pending_loud.clear()
                continue

            await self._send_json(
                ws, {"type": "mic-audio-data", "audio": chunk.tolist()}
            )
            if loud:
                quiet_run = 0
            else:
                quiet_run += 1
                if quiet_run >= silence_end:
                    await self._send_json(ws, {"type": "mic-audio-end"})
                    print(
                        "[BRIDGE] Sent mic-audio-end (ASR). Watch for [YOU] transcription "
                        "then [AI] reply / audio."
                    )
                    in_utterance = False
                    quiet_run = 0
                    pending_loud.clear()

    # -- Text input (text mode) ----------------------------------------------

    async def _text_input_loop(self, ws: Any) -> None:
        """Read text from stdin and send to server (bypasses ASR)."""
        print("[BRIDGE] Text mode active. Type messages and press Enter.")
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            text = line.strip()
            if not text:
                continue
            await self._send_json(
                ws,
                {
                    "type": "text-input",
                    "text": text,
                },
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI flags and run the async bridge."""
    if websockets is None:
        sys.exit("Missing dependency: uv add websockets  (or: uv sync --extra reachy)")
    parser = argparse.ArgumentParser(
        description="Bridge between Open-LLM-VTuber and Reachy Mini Lite"
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:12393/client-ws",
        help="Open-LLM-VTuber WebSocket URL (default: ws://localhost:12393/client-ws)",
    )
    parser.add_argument(
        "--usb",
        action="store_true",
        help="Connect to an existing Reachy daemon (robot or manual sim); "
        "omit for auto-spawned MuJoCo simulation",
    )
    parser.add_argument(
        "--reachy-host",
        default="",
        help="Reachy daemon hostname or IP (optional; enables network mode)",
    )
    parser.add_argument(
        "--reachy-daemon-port",
        type=int,
        default=8000,
        help="HTTP port of the Reachy Mini daemon (default: 8000)",
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Use keyboard text input instead of microphone",
    )
    parser.add_argument(
        "--keep-sim-daemon",
        action="store_true",
        help="When using local MuJoCo sim, do not stop reachy-mini-daemon --sim on exit",
    )
    args = parser.parse_args()

    host = args.reachy_host.strip() or None
    reachy, manage_sim_daemon = connect_reachy(
        use_usb=args.usb,
        reachy_host=host,
        reachy_daemon_port=args.reachy_daemon_port,
    )

    if reachy is not None and HAS_SOUNDDEVICE and not args.text_mode:
        release = getattr(reachy, "release_media", None)
        if callable(release):
            try:
                release()
                logger.info("Released Reachy media so the local mic can be used.")
            except Exception:
                logger.exception(
                    "reachy.release_media() failed (mic may not work with daemon)"
                )

    prime_reachy_for_bridge(reachy)

    bridge = ReachyBridge(
        server_url=args.server,
        reachy=reachy,
        text_mode=args.text_mode,
    )

    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        stop_sounddevice()
        try:
            bridge._shutdown.set()
        except Exception:
            pass
        print("\n[BRIDGE] Shutting down.")
    finally:
        stop_sounddevice()
        shutdown_reachy(reachy)
        if manage_sim_daemon and not args.keep_sim_daemon:
            terminate_reachy_sim_daemons_bounded()


if __name__ == "__main__":
    main()
