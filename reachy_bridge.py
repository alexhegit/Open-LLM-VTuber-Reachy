"""
Reachy Mini ↔ Open-LLM-VTuber Bridge Client
=============================================
WebSocket client that connects to Open-LLM-VTuber's /client-ws endpoint,
receives audio + emotion expressions, and drives a Reachy Mini Lite robot.

Usage:
    uv run reachy_bridge.py                  # MuJoCo simulation
    uv run reachy_bridge.py --usb            # USB connected robot
    uv run reachy_bridge.py --text-mode      # Keyboard text input (no mic)
    uv run reachy_bridge.py --server ws://192.168.1.100:12393/client-ws

Requires: websockets, numpy, sounddevice (install with:
    uv add websockets sounddevice
)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import struct
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

try:
    import websockets
except ImportError:
    sys.exit("Missing dependency: uv add websockets")

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
#
# Replace the bodies of reachy_joy(), reachy_sadness(), etc. with your
# ReachyMiniChat emotion_to_action() calls or direct SDK poses.

SAMPLE_RATE = 16000  # Open-LLM-VTuber expects 16 kHz


def reachy_neutral(reachy: Any) -> None:
    """Return to neutral pose."""
    # TODO: Replace with your ReachyMiniChat mapping
    # e.g. emotion_to_action(reachy, "neutral")
    pass


def reachy_sadness(reachy: Any) -> None:
    """Express sadness (head down, slow movement)."""
    # TODO: Replace with your ReachyMiniChat mapping
    pass


def reachy_anger(reachy: Any) -> None:
    """Express anger (sharp head forward)."""
    # TODO: Replace with your ReachyMiniChat mapping
    pass


def reachy_joy(reachy: Any) -> None:
    """Express joy (head up, lively movement)."""
    # TODO: Replace with your ReachyMiniChat mapping
    pass


# Maps expression index → action function (mao_pro model_dict.json indices)
EXPRESSION_ACTIONS: dict[int, callable] = {
    0: reachy_neutral,
    1: reachy_sadness,
    2: reachy_anger,
    3: reachy_joy,
}

# Maps emotion name strings → action functions (for messages that send names)
EMOTION_NAME_ACTIONS: dict[str, callable] = {
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

    Returns:
        (samples, sample_rate) where samples is float32 mono.
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
    audio /= 32768.0  # normalize to [-1, 1]
    return audio, sr


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    """Play audio through the default output device."""
    if not HAS_SOUNDDEVICE:
        print("[WARN] sounddevice not installed, skipping audio playback")
        return
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def record_chunk(duration_s: float = 0.1) -> np.ndarray:
    """Record a short audio chunk from the default mic.

    Returns float32 samples at SAMPLE_RATE.
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

def connect_reachy(use_usb: bool = False) -> Any:
    """Connect to the Reachy Mini Lite robot.

    Args:
        use_usb: If True, connect via USB. Otherwise use MuJoCo simulation.

    Returns:
        Reachy robot instance.
    """
    try:
        from reachy_mini import ReachyMini

        if use_usb:
            robot = ReachyMini(port="/dev/ttyACM0")
        else:
            robot = ReachyMini(simulated=True)
        print(f"[REACHY] Connected ({'USB' if use_usb else 'MuJoCo simulation'})")
        return robot
    except ImportError:
        print("[WARN] reachy-mini SDK not found. Emotions will be logged only.")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to connect to Reachy: {e}")
        return None


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

    async def run(self) -> None:
        """Main loop: connect, listen, and drive the robot."""
        print(f"[BRIDGE] Connecting to {self.server_url} ...")
        async with websockets.connect(self.server_url) as ws:
            print("[BRIDGE] Connected!")
            if self.text_mode:
                await asyncio.gather(
                    self._receive_loop(ws),
                    self._text_input_loop(ws),
                )
            else:
                await asyncio.gather(
                    self._receive_loop(ws),
                    self._mic_input_loop(ws),
                )

    # -- Receiving from server -----------------------------------------------

    async def _receive_loop(self, ws: Any) -> None:
        """Listen for messages from Open-LLM-VTuber."""
        async for raw in ws:
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio":
                await self._handle_audio(msg)
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

    async def _handle_audio(self, msg: dict) -> None:
        """Decode audio, play it, and trigger emotion actions."""
        b64_audio = msg.get("audio")
        actions = msg.get("actions") or {}
        expressions = actions.get("expressions") or []
        display_text = msg.get("display_text") or {}
        text = display_text.get("text", "")

        # Show text
        if text:
            print(f"[AI] {text}")

        # Trigger emotion actions (non-blocking, fire-and-forget)
        if expressions:
            self._apply_expressions(expressions)

        # Decode and play audio
        if b64_audio:
            self._playing.clear()
            try:
                audio, sr = decode_wav_to_samples(b64_audio)
                await asyncio.get_event_loop().run_in_executor(
                    None, play_audio, audio, sr
                )
            except Exception as e:
                print(f"[ERROR] Audio playback failed: {e}")
            finally:
                self._playing.set()

                # Notify server that playback is complete
                # (not strictly required for basic usage, but polite)

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
            print("[BRIDGE] AI is thinking...")
        elif command == "conversation-chain-end":
            print("[BRIDGE] AI response complete.")
        elif command == "interrupt":
            print("[BRIDGE] Interruption detected.")

    # -- Microphone input (auto mode) ----------------------------------------

    async def _mic_input_loop(self, ws: Any) -> None:
        """Continuously record mic and send audio chunks to server."""
        print("[BRIDGE] Mic input active. Start speaking...")
        while True:
            # Wait for any current playback to finish
            await self._playing.wait()

            chunk = await asyncio.get_event_loop().run_in_executor(
                None, record_chunk, 0.1
            )
            if chunk.size == 0:
                await asyncio.sleep(0.1)
                continue

            # Send chunk as float32 list
            await ws.send(json.dumps({
                "type": "mic-audio-data",
                "audio": chunk.tolist(),
            }))

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
            await ws.send(json.dumps({
                "type": "text-input",
                "text": text,
            }))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
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
        help="Connect to Reachy via USB instead of MuJoCo simulation",
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Use keyboard text input instead of microphone",
    )
    args = parser.parse_args()

    reachy = connect_reachy(use_usb=args.usb)
    bridge = ReachyBridge(
        server_url=args.server,
        reachy=reachy,
        text_mode=args.text_mode,
    )

    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        print("\n[BRIDGE] Shutting down.")
    finally:
        if reachy:
            try:
                reachy.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
