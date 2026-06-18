# Reachy Mini + Open-LLM-VTuber 集成方案

## 架构

```
┌──────────────────────────┐        WebSocket         ┌──────────────────────┐
│   Open-LLM-VTuber 服务端  │ ◄──────────────────────► │   桥梁客户端 (reachy_bridge.py) │
│   ASR → LLM → TTS        │   ws://localhost:12393   │                                 │
│   表情标签 + 音频 + 文本    │   /client-ws            │   消息翻译 + 路由 + 情绪映射      │
└──────────────────────────┘                          └──────────┬───────────┘
                                                                 │
                                                        可选: OLV_REACHY_EMOTION_CALLABLE
                                                                 │
                                                      ┌──────────▼───────────┐
                                                      │   reachy-mini SDK     │
                                                      │   goto_target(head)   │
                                                      └──────────┬───────────┘
                                                                 │
                                                    ┌────────────▼────────────┐
                                                    │  Reachy Mini Lite         │
                                                    │  (daemon: MuJoCo / USB)   │
                                                    └─────────────────────────┘
```

## 三个对接点

### 1. 情绪 → 动作

Open-LLM-VTuber 在 `audio` 消息中通过 `actions.expressions` 发送整数索引（来自 `model_dict.json` 的 `emotionMap`）。

mao_pro 模型的映射：

| 索引 | 情绪 |
|------|------|
| 0 | neutral |
| 1 | fear, sadness |
| 2 | anger, disgust |
| 3 | joy, smirk, surprise |

桥梁客户端将索引翻译为情绪名称，调用内置 `goto_target(create_head_pose(...))`，或通过环境变量
`OLV_REACHY_EMOTION_CALLABLE=my.module:func` 调用你的 `emotion_to_action(reachy, emotion)`。

### 2. 语音输出

服务端发送 base64 编码的 WAV 音频（16kHz, 16-bit PCM, mono），通过 `audio` 消息的 `audio` 字段。桥梁客户端解码后通过 `sounddevice` 播放，或调用 Reachy SDK 扬声器。

### 3. 语音输入

用 `sounddevice` 录制 16kHz float32 音频，通过 WebSocket 发送 `mic-audio-data`（`{"audio": [float32 samples]}`），录音结束后发 `mic-audio-end` 触发 ASR → LLM → TTS 管道。

## WebSocket 协议要点

### 客户端 → 服务端

```json
// 发送麦克风音频块
{"type": "mic-audio-data", "audio": [0.1, -0.2, ...]}

// 录音结束，触发处理
{"type": "mic-audio-end"}

// 文本输入（跳过 ASR）
{"type": "text-input", "text": "你好"}
```

### 服务端 → 客户端

```json
// AI 回复（音频 + 文本 + 表情）
{
  "type": "audio",
  "audio": "base64编码的WAV",
  "volumes": [0.8, 0.9, ...],
  "display_text": {"text": "你好啊！", "name": "Mao"},
  "actions": {"expressions": [3]}
}

// 控制信号
{"type": "control", "text": "conversation-chain-start"}
{"type": "control", "text": "conversation-chain-end"}
```

## 已完成

- [x] `reachy_bridge.py`：桥梁客户端；MuJoCo 自拉起 daemon / `--usb` 连已有 daemon；麦克风与文本模式；内置情绪头姿 + `OLV_REACHY_EMOTION_CALLABLE` 可插拔钩子
- [x] `pyproject.toml`：torch 从核心依赖移除，支持 CUDA/ROCm/CPU 独立安装
- [x] `AGENTS.md`：更新了 GPU 安装说明和桥梁客户端文档
- [x] 协议调研：完整梳理了 WebSocket 消息格式

## 待完成

- [ ] 安装 ROCm 版 torch（`uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2`）
- [x] `reachy_bridge.py`：内置 `create_head_pose` + `goto_target` 情绪动作；可选 `OLV_REACHY_EMOTION_CALLABLE=my.mod:fn` 对接自定义 `emotion_to_action(reachy, emotion)`；连接逻辑对齐当前 `reachy-mini` daemon（仿真 spawn / USB 连已有 daemon）
- [ ] 端到端测试：启动服务端 → 启动桥梁 → 说话 → 验证机器人动作和音频播放

## 运行方式

```bash
# 1. 安装依赖
uv sync
uv sync --extra reachy
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
# reachy-mini[mujoco] 已含于 --extra reachy（见 REACHY.md / AGENTS.md）

# 2. 启动服务端
uv run run_server.py

# 3. 启动桥梁
uv run reachy_bridge.py                      # 自动拉起 MuJoCo 仿真 daemon
uv run reachy_bridge.py --usb                # 连接本机/已在跑的硬件 daemon
uv run reachy_bridge.py --text-mode          # 键盘输入
```
