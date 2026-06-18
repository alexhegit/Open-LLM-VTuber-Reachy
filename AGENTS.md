# AGENTS.md

## Quick Commands

- **Lint**: `uv run ruff check .`
- **Tests**: `uv sync --group dev` then `uv run pytest` (combine with Reachy: `uv sync --group dev --extra reachy`)
- **Format**: `uv run ruff format .`
- **Pre-commit**: `pre-commit run --all-files`
- **Run server**: `uv run run_server.py`
- **Run with verbose logs**: `uv run run_server.py --verbose`
- **Update**: `uv run upgrade.py`

## Package Management

Always use `uv` (not pip). Use `uv run`, `uv sync`, `uv add`, `uv remove`.

After adding a dependency to `pyproject.toml`, also add it to `requirements.txt`.

## Code Style (enforced by Ruff + pre-commit)

- Target Python 3.10+. Use `|` for unions (not `Optional`), built-in generics (not `List`, `Dict`).
- All public functions/classes/methods need Google-style docstrings with `Args:` and `Returns:` sections.
- Logging via `loguru` only. Log messages in English.
- No comments unless asked.
- Lint passes `ruff check --exit-non-zero-on-fix` in CI (pre-commit + GitHub Actions).

## Architecture

- **Entrypoint**: `run_server.py` starts `WebSocketServer` (FastAPI + Uvicorn).
- **Core source**: `src/open_llm_vtuber/` — single package, no sub-packages split across repos.
- **Config**: `conf.yaml` (user) generated from `config_templates/conf.default.yaml` or `conf.ZH.default.yaml`. Validated by Pydantic models in `src/open_llm_vtuber/config_manager/`.
- **Frontend**: Git submodule at `frontend/` (React app from separate repo `Open-LLM-VTuber-Web`). Do NOT edit directly — use `git restore frontend` if corrupted.
- **Live2D models**: `live2d-models/` directory; model names must match entries in `model_dict.json`.
- **Character configs**: `characters/` directory (YAML).
- **Prompts**: `prompts/` directory.

## Config Changes

When modifying configuration structure, update **both** `config_templates/conf.default.yaml` AND `config_templates/conf.ZH.default.yaml`, plus the Pydantic models in `src/open_llm_vtuber/config_manager/`.

## Key Gotchas

- Python version: `3.10` (pinned in `.python-version`). Requires `>=3.10,<3.13`.
- `doc/` directory is deprecated — do not add documentation there.
- Models downloaded via ModelScope/HuggingFace land in `models/` (env vars `HF_HOME` and `MODELSCOPE_CACHE` are set in `run_server.py`).
- The `scripts/run_bilibili_live.py` has an `E402` lint exception (ruff per-file-ignore in `pyproject.toml`).
- HTTPS required for remote access (browser mic API needs secure context). See README for reverse proxy setup.
- **torch is NOT in `pyproject.toml` core deps** — must be installed separately per GPU type (see below).

## GPU / PyTorch Installation

torch is intentionally excluded from `pyproject.toml` because CUDA, ROCm, and CPU variants conflict. Install the right one for your hardware **after** `uv sync`:

```bash
# NVIDIA CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# AMD ROCm (HIP)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2

# CPU only
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

The code uses `torch.cuda.is_available()` for device detection — this works for both CUDA and ROCm (ROCm emulates the CUDA API). No code changes needed when switching GPU types.

## Reachy Mini Bridge Client (`reachy_bridge.py`)

Standalone WebSocket client that replaces the browser frontend — connects to Open-LLM-VTuber's `/client-ws` endpoint and drives a Reachy Mini Lite robot instead of a Live2D avatar.

**Usage:**
```bash
uv run reachy_bridge.py                      # Spawns local MuJoCo sim daemon, mic input
uv run reachy_bridge.py --usb              # Connect to an already-running Reachy daemon
uv run reachy_bridge.py --reachy-host IP   # Remote daemon (network mode)
uv run reachy_bridge.py --text-mode        # Keyboard input (no mic needed)
uv run reachy_bridge.py --keep-sim-daemon  # Leave sim daemon running after bridge exits
uv run reachy_bridge.py --server ws://HOST:PORT/client-ws
```

**Install bridge deps:** `uv sync --extra reachy` installs **`reachy-mini[mujoco]`**, **`sounddevice`**, and **`websockets`**. See **[REACHY.md](REACHY.md)** for setup, networking notes, and emotion hooks.

**Custom emotion mapping:** set `OLV_REACHY_EMOTION_CALLABLE=my.module:function` where `function(reachy, emotion: str)` receives canonical labels `neutral`, `sadness`, `anger`, `joy`. If unset, the bridge uses built-in head poses via `reachy_mini.utils.create_head_pose` and `goto_target`.

**Key protocol facts for modifying the bridge:**
- Server audio: base64-encoded WAV, 16kHz, 16-bit PCM, mono
- Client mic audio: `mic-audio-data` chunks, then **`mic-audio-end`** after silence to trigger ASR. After TTS, the bridge sends **`frontend-playback-complete`** when local playback finishes (required by the server).
- Text input (bypasses ASR): `{"type": "text-input", "text": "..."}`
- Emotions arrive in `audio` messages under `actions.expressions` as integer indices (mao_pro: 0=neutral, 1=sadness/fear, 2=anger/disgust, 3=joy/surprise)
