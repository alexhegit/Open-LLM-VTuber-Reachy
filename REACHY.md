# Reachy Mini integration

[简体中文摘要](#简体中文摘要) · [English sections below](#dependencies)

## 简体中文摘要

本仓库通过根目录的 **`reachy_bridge.py`** 把 Open-LLM-VTuber 的 **`/client-ws`** 语音与表情流接到 **Pollen Reachy Mini Lite**（MuJoCo 仿真或真机）。安装请执行 **`uv sync --extra reachy`**（包含官方 **`reachy-mini[mujoco]`**、麦克风 **`sounddevice`**、**`websockets`**）。服务端默认 **`tts_model: edge_tts`**（微软 Edge 语音，需联网）。麦克风模式会在检测到说话后发送 ``mic-audio-data``，并在停顿约 1.5 秒后发送 ``mic-audio-end`` 才会触发服务端 ASR（与网页端协议一致）。环境变量 **`OLV_REACHY_EMOTION_CALLABLE`**、**`OLV_BRIDGE_SPEECH_RMS`** 等见 **`REACHY.md`** 英文小节与 **`AGENTS.md`**。

---

This fork adds a **WebSocket bridge client** (`reachy_bridge.py`) that connects to Open-LLM-VTuber’s `/client-ws` endpoint and drives a **Pollen Robotics Reachy Mini Lite** (simulation or hardware) instead of the browser Live2D UI.

## Dependencies

Install everything needed for the bridge (audio, WebSockets, and the official **reachy-mini** SDK with the **MuJoCo** extra for local simulation):

```bash
uv sync --extra reachy
```

The `reachy` optional dependency group in `pyproject.toml` includes:

- `reachy-mini[mujoco]` — Pollen SDK plus MuJoCo for the default simulated robot path
- `sounddevice` — microphone capture
- `websockets` — async WebSocket client

The main project pins **NumPy 2.x** and a compatible **SciPy** so it resolves cleanly with current `reachy-mini` releases on PyPI. If you only run the server (no bridge), `uv sync` without `--extra reachy` still uses those pins.

The checked-in **`requirements.txt`** is generated with `uv export --frozen` for the **default** dependency set (server install). It does **not** list optional `reachy` packages; use `uv sync --extra reachy` or `pip install -e ".[reachy]"` when you need the bridge and SDK.

### PyTorch

`torch` is still **not** part of the default dependency set (GPU-specific wheels). Install it separately as described in `AGENTS.md` / `README.md` if your ASR or other components need it.

## Run the bridge

Start the Open-LLM-VTuber server as usual (`uv run run_server.py`), then in another terminal:

```bash
# Local MuJoCo simulation (spawns a Reachy daemon) + microphone
uv run reachy_bridge.py

# Connect to an already-running daemon (e.g. real robot over USB)
uv run reachy_bridge.py --usb

# Remote daemon
uv run reachy_bridge.py --reachy-host <IP>

# Text input only (no microphone)
uv run reachy_bridge.py --text-mode

# Custom server URL (use 127.0.0.1 if localhost WebSocket fails behind proxy)
uv run reachy_bridge.py --server ws://127.0.0.1:12393/client-ws
```

See `reachy_bridge.py --help` for all flags.

## Latency: long “Thinking…” before first voice (text or mic)

This is **usually expected**, not a bridge bug.

1. **“Thinking…” is immediate** — the server sends `conversation-chain-start` and a placeholder `full-text` as soon as your message is accepted (`send_conversation_start_signals` in `conversation_utils.py`). It does **not** mean the LLM has finished.
2. **First audible TTS waits for a “chunk” of reply** — the agent streams tokens, but the **sentence divider** only hands the first chunk to TTS after it sees a **comma** (when `faster_first_response` is true) or **sentence-ending punctuation**, or after the stream ends. Until then, the UI can stay on “Thinking…” while the model is still generating.
3. **Reasoning / `think`-tagged segments** — if the model emits long internal-reasoning blocks first (parsed as `think` tags), those are **not spoken** (`tts_filter` skips TTS for them). The first voice only starts after **normal** speakable text is segmented — which can feel like a very long “think” phase.
4. **TTS itself** — the first `audio` message is sent only after the first phrase is synthesized; slow engines (large local models, cloud cold start) add seconds on top of LLM time.

**What you can tune (server / config, not the bridge):**

- Use a **smaller or faster LLM**, local **GPU**, or an API with lower latency; cold-start first request is often slower.
- Prefer a **fast TTS** (e.g. **Edge TTS** in config) if you need snappy playback.
- In `conf.yaml` / character agent settings, keep **`faster_first_response: true`** (default) so the first split can happen at the **first comma**; **`segment_method`** is `pysbd` by default (`config_templates/conf.default.yaml`).
- If the model supports it, **disable or shorten “reasoning”** so less text is hidden inside `think`-style blocks before the answer.
- Run **`uv run run_server.py --verbose`** and watch logs: time from “User input” to first TTS / `audio` indicates whether the delay is **LLM** vs **TTS**.

## Microphone protocol (important)

The server **buffers** `mic-audio-data` and only runs ASR + LLM + TTS after a **`mic-audio-end`** message (see `websocket_handler.py`). The bridge uses a simple **energy detector**: after several consecutive loud chunks it streams audio; when **~1.5 s of silence** follows, it sends `mic-audio-end`. If your room is noisy or very quiet, tune:

- `OLV_BRIDGE_SPEECH_RMS` — RMS threshold; **if unset**, the bridge **calibrates ~1.8s** from ambient noise (25th percentile × 4.5, clamped). Set explicitly if calibration is wrong for your mic.
- `OLV_BRIDGE_SILENCE_CHUNKS` — number of quiet 0.1 s chunks before end-of-utterance (default `15` ≈ 1.5 s).
- `OLV_BRIDGE_MIN_SPEECH_CHUNKS` — consecutive loud chunks required to start an utterance (default `3`).

Debug / exit tuning:

- `OLV_BRIDGE_DEBUG_WS=1` — log every WebSocket message type sent/received (verify `mic-audio-end` and server `audio` messages).
- `OLV_BRIDGE_DEBUG_MIC=1` — print live RMS vs threshold while waiting for speech.
- `OLV_REACHY_FULL_EXIT=1` — on exit, also run the SDK ``__exit__`` (closes media); default is **disconnect-only** so Ctrl+C does not hang if the sim daemon is already dead.

After each reply, the server waits for **`frontend-playback-complete`** once synthesis is done; the bridge sends this after local speaker playback (same obligation as the web client). If that message is missing, the server still ends the turn so the next utterance can be processed.

### Server-side audio gate (optional)

Set **`OLV_MIN_MIC_SAMPLES`** on the server process to change the minimum float32 samples required before ASR runs after `mic-audio-end` (default **2400**, about 150 ms at 16 kHz). Shorter buffers are ignored with an error message to the client.

## Emotion mapping

The server sends expression indices; the bridge maps them to head poses using `reachy_mini.utils.create_head_pose` and `goto_target` when the SDK is available.

To plug in your own behaviour, set:

```bash
export OLV_REACHY_EMOTION_CALLABLE=my_package.module:emotion_to_action
```

The callable must accept `(reachy, emotion: str)` where `emotion` is one of: `neutral`, `sadness`, `anger`, `joy`.

## MuJoCo sim: first connect and daemon cleanup

The Pollen SDK starts `reachy-mini-daemon --sim` with `subprocess.Popen` and then connects immediately; the daemon often is not listening on port 8000 yet, which used to surface as `Connection refused` while the daemon later appeared in logs. This bridge **waits for TCP on port 8000 and retries** once, and by default **stops `reachy-mini-daemon --sim` on exit** (Ctrl+C) so port 8000 is not left occupied. Use **`--keep-sim-daemon`** if you want the sim daemon to keep running after the bridge exits.

If you see **GStreamer / webrtc** errors in daemon logs, the sim path uses **`media_backend="no_media"`** in the client to avoid requiring optional WebRTC plugins for basic head motion.

## Networking and proxies

If the WebSocket handshake fails with empty or invalid responses, try:

- `127.0.0.1` instead of `localhost`
- Unset `HTTP_PROXY` / `HTTPS_PROXY` for the bridge process, or add the server host to `NO_PROXY`

## Further reading

- `AGENTS.md` — short command reference for this repository
- `reachy_bridge.py` module docstring — protocol summary (audio format, message types)
