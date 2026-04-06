# Voice

语音服务。实现唤醒词检测（KWS）、语音活动检测（VAD）、自动语音识别（ASR）和文字转语音（TTS），向 Orchestrator 发送结构化指令。

## 架构

```
Mic (arecord) → KWS (OpenWakeWord) → VAD (FSMN) → ASR (Paraformer)
                                                  ↓
                                         Intent Inference
                                                  ↓
                              Orchestrator ◀──task_cmd
                              Orchestrator ──task_ack──▶ [voice]
                              Orchestrator ──tts_event──▶ TTS (Piper) → Speaker
```

## 目录结构

```
Voice/
├── kws/
│   ├── wake_nihao_xiaoche_v2.onnx   # 唤醒词模型（"你好小车"）
│   └── stop_smallcar_v1.onnx        # 停止词模型
├── tts/
│   └── zh_CN-huayan-x_low/          # Piper TTS 中文模型
├── tts_cache/                        # 预合成 TTS 音频缓存
├── scripts/
│   └── sc171_vad_asr_kws_tts_dualthread_concise.py  # 主程序
├── thirdparty/                       # 第三方依赖（不纳入版本控制）
│   ├── fsmn_vad/
│   ├── paraformer_asr/
│   ├── paraformer_asr_online/
│   └── openwakeword_models/
└── logs/ / pids/
```

## 组件

| 组件 | 实现 | 说明 |
|------|------|------|
| KWS | OpenWakeWord + ONNX | 唤醒词"你好小车"，停止词"停止" |
| VAD | FSMN-VAD（funasr_onnx） | 语音端点检测，裁剪有效语音段 |
| ASR | Paraformer（funasr_onnx） | 中文离线识别，双线程异步处理 |
| TTS | Piper（`zh_CN-huayan-x_low`） | 中文语音合成，支持缓存复用 |

## 运行

```bash
# 通过全栈脚本启动（推荐）
./start_robot_stack.sh

# 手动启动（需 conda env: asr）
conda activate asr
cd /home/aidlux/2026/Voice_service
python3 -m voice_service.app.main
```

## IPC 协议

| 方向 | 消息类型 | 地址 |
|------|----------|------|
| 发送 | `task_cmd` | `127.0.0.1:9001` |
| 接收 | `task_ack` | `127.0.0.1:9012` |
| 接收 | `tts_event` | `127.0.0.1:9011` |

`task_cmd` 输出格式（JSONL stdout）：

```json
{"ts": 1234567890.0, "intent": "FIND", "target": "apple", "confidence": 0.75, "text": "找苹果"}
{"ts": 1234567890.0, "intent": "STOP", "confidence": 0.98, "source": "stop_kws"}
{"ts": 1234567890.0, "intent": "RETURN", "confidence": 0.85, "text": "回来"}
```

支持的 intent：`FIND`（目标搜索）、`STOP`（立即停止）、`RETURN`（返回原点）

## 指令规则

默认指令映射（可通过 `--commands_json` 覆盖）：

| intent | 触发关键词 |
|--------|-----------|
| STOP | 停止、停下、别动、取消、危险 |
| RETURN | 回来、返回、回去 |
| FIND cup | 水杯、杯子 |
| FIND apple | 苹果 |
| FIND keys | 钥匙 |

## 依赖安装

```bash
conda activate asr
pip install funasr_onnx onnxruntime piper-tts soundfile numpy
```
