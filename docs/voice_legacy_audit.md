# Voice Legacy Audit Report

This report summarizes the audit of the legacy Voice package (`Voice_legacy/Voice`) and outlines the findings, migration feasibility, and integration points for the new Voice Gateway in the `embedded_com` repository.

## 1. Voice_legacy 的真实目录结构 (Actual Directory Structure)

The actual legacy Voice directory is located at `d:\55495\workspace\Voice` and contains the following structure:
```text
d:\55495\workspace\Voice/
├─ ONLINE_ASR_MIGRATION_NOTES.md
├─ README.md
├─ start_voice_asr.sh
├─ config/
│  ├─ commands.json
│  └─ commands.example.json
├─ kws/
│  ├─ wake_nihao_xiaoche_v2.onnx
│  ├─ wake_nihao_xiaoche.onnx
│  ├─ wake_nihao_xiaoche_clean.onnx
│  ├─ stop_smallcar_v1.onnx
│  ├─ stop_xiaochetingzhi_v1.onnx
│  └─ stop_smallcar.onnx
├─ tts/
│  └─ zh_CN-huayan-x_low/
│     ├─ zh_CN-huayan-x_low.onnx
│     └─ zh_CN-huayan-x_low.onnx.json
├─ scripts/
│  ├─ check_dim.py
│  ├─ inspect_funasr_online.py
│  ├─ test_online_model_minimal.py
│  ├─ sc171_vad_asr_kws_tts_dualthread.py
│  └─ sc171_vad_asr_kws_tts_dualthread_concise.py
└─ voice_service/
   ├─ __init__.py
   ├─ app/
   │  ├─ __init__.py
   │  └─ main.py
   ├─ config/
   │  ├─ __init__.py
   │  ├─ board_config.py
   │  └─ schema.py
   ├─ ipc/
   │  ├─ __init__.py
   │  ├─ protocol.py
   │  └─ transport.py
   └─ runtime/
      ├─ __init__.py
      ├─ asr_engine.py
      ├─ commands.py
      ├─ common.py
      ├─ kws_engine.py
      ├─ mic_stream.py
      ├─ state.py
      ├─ tts_engine.py
      └─ workers.py
```

## 2. 当前主入口 (Main Entry Point)

- **Entry Script**: `voice_service/app/main.py`
  - Loads configuration from `voice_service.config.CONFIG`.
  - Runs `run_voice_service(CONFIG)` from `voice_service/runtime/service.py`.
- **Main Service**: `voice_service/runtime/service.py`
  - Spawns two daemon threads: `AudioKWSWorker` (mic capture + KWS/STOP detection) and `ASRDecisionWorker` (VAD, ASR, command interpretation, TTS).
  - Starts listeners for task acknowledgments (`JsonlInboundListener`) and TTS events if enabled.

## 3. KWS 实现 (KWS Implementation)

- **Engine**: `voice_service/runtime/kws_engine.py` (`FlexibleWakeWord`)
- Uses `openwakeword` library. Features are extracted via `AudioFeatures` and fed into a wake-word `.onnx` or `.tflite` model.
- Defaults to CPU-based inference provider on CPU executing `.onnx` models (`onnxruntime`) or `.tflite` models (`tflite_runtime`).
- Supports standard (shape `[1, 96, N]`) and transposed (shape `[1, N, 96]`) feature layouts.
- VAD gating (`VADProcessor` or `openwakeword.VAD`) can be enabled optionally to filter non-speech signals before wake word classification.

## 4. STOP KWS 实现 (STOP KWS Implementation)

- Also managed by `FlexibleWakeWord`.
- Uses a separate model specifically for the "STOP" keyword (e.g. `stop_smallcar_v1.onnx`).
- Under `AudioKWSWorker.run`, if `stop_tflite` (the stop model path) and `stop_key` are set, and state allows (`self.rt.can_trigger_stop()`), KWS classification runs.
- Triggering "STOP" bumps command epoch, discards queued audio, clears pending utterances, resets the pipeline state, and immediately sends a high-priority `STOP` task command.

## 5. 音频采集 (Audio Capture)

- **Module**: `voice_service/runtime/mic_stream.py` (`RawMicStream`)
- Spawns `arecord` as a subprocess to record mono, 16kHz, 16-bit PCM audio from the configured device (e.g., `plughw:1,0`).
- Reads data via stdout using select and non-blocking I/O.
- Includes automatic restart and recovery on EOF or timeouts.
- **Limitation**: Windows is NOT supported due to reliance on `arecord` and `select` with Unix file descriptors. Must be mocked or bypassed on Windows dev mode.

## 6. VAD 与分段 (VAD and Segmentation)

- **Module**: `voice_service/runtime/asr_engine.py` (`VADProcessor` & `AudioCommandPipeline`)
- In **offline** mode: Uses FunASR FSMN-VAD (`funasr_onnx.Fsmn_vad`) to segment captured voice audio. It extracts the best speech segment from the audio file using a temporary WAV wrapper.
- In **online** mode: Uses energy-based segmenting thresholds (`energy_th`, `start_frames`, `end_frames`, `pre_frames`, `max_frames`) inside `AudioKWSWorker` to group raw audio chunks into a session before sending them to the online ASR encoder chunk-by-chunk.

## 7. ASR online/offline (ASR Online/Offline)

- **Module**: `voice_service/runtime/asr_engine.py` (`OfflineASREngine` & `OnlineASREngine`)
- **Offline ASR**: Uses FunASR Paraformer Offline model (`funasr_onnx.Paraformer`) to transcribe the entire segmented audio chunk at once.
- **Online ASR**: Uses FunASR Paraformer Online model (`funasr_onnx.paraformer_online_bin.Paraformer`) to continuously feed audio chunks (`CHUNK` events) to an active session, displaying partial results (`partial_text`), and finishing transcription on `FINAL` event.

## 8. Intent Parser (Intent Parser)

- **Module**: `voice_service/runtime/commands.py` (`CommandInterpreter`)
- Loads command rules from a JSON config file (`Voice/config/commands.json` by default).
- Compares normalized Chinese text against keyword rules:
  - Match stop keywords -> `STOP` intent.
  - Match return keywords -> `RETURN` intent.
  - Match target object keywords -> `FIND` intent, mapping the spoken name to the exact class ID target (e.g., "苹果" -> `apple`).
  - No match -> `REJECT`.

## 9. TaskCmd (Task Command Construction)

- **Protocol**: `voice_service/ipc/protocol.py` (`build_task_cmd`)
- Formulates a dictionary containing:
  - `ts`: timestamp
  - `type`: `"task_cmd"`
  - `intent`: `FIND` / `RETURN` / `STOP`
  - `confidence`: confidence score
  - `cmd_id`, `session_id`, `epoch`
  - `source`: `"voice"`
  - `target` (if `intent` is `FIND`)
  - Optional metadata: `text`, `raw_text`, `high_priority`

## 10. TaskAck (Task Ack Receiving)

- **Protocol**: `voice_service/ipc/protocol.py` (`normalize_task_ack`)
- Parses incoming task acknowledgment payloads:
  - `ts`, `type`: `"task_ack"`
  - `cmd_id`, `session_id`, `epoch`
  - `accepted`: boolean indicating whether the Orchestrator accepted the command
  - `state`: orchestrator active state
  - `reason`: rejection reason if not accepted

## 11. TTS (Text-to-Speech)

- **Module**: `voice_service/runtime/tts_engine.py` (`PiperTTS` & `ThreadSafeTTS`)
- Uses Piper ONNX (`piper.voice.PiperVoice`) to synthesize speech.
- Falls back to piper command line interface if `PiperVoice` is not imported.
- Caches generated speech WAV files to disk (`tts_cache/`) and plays them using `aplay -q` (configurable).
- Bypasses playback if local speaker is disabled.

## 12. 状态保护 (State Protection & Guard Mechanisms)

- **Module**: `voice_service/runtime/state.py` (`RuntimeState`)
- **Epoch Management**: Keeps track of `command_epoch` to discard stale ASR results and incoming late ACKs. Bumps epoch immediately upon detecting "STOP".
- **STOP Priority**: STOP commands skip queueing, discard all pending utterances/ASR streams, and bypass standard mute/guard timers.
- **Mute Protection**: Mutes voice input during local TTS playback (`post_tts_mute_secs`) or immediately after wake trigger (`post_wake_mute_secs`).
- **Post-Stop Guard**: Prevents re-triggering KWS for `stop_guard_secs` and blocks repeated stop commands for `stop_repeat_block_secs`.

## 13. 日志 (Logging)

- **Logging Utilities**: `voice_service/runtime/common.py`
  - Structured JSON logging (`jlog`).
  - Event-specific files: `timeline.jsonl`, `ipc.jsonl`, `state.jsonl`, `stop_trace.jsonl` under active run directories.
  - Console-friendly concise logs for normal operation.

## 14. 可直接迁移部分 (Directly Migratable Parts)

- `voice_service/runtime/kws_engine.py` (FlexibleWakeWord)
- `voice_service/runtime/tts_engine.py` (PiperTTS, ThreadSafeTTS)
- `voice_service/runtime/state.py` (RuntimeState)
- `voice_service/runtime/commands.py` (CommandInterpreter)
- `voice_service/runtime/asr_engine.py` (AudioCommandPipeline, VADProcessor, ASR engines)
- `voice_service/runtime/workers.py` (AudioKWSWorker, ASRDecisionWorker, TTSEventListenerFactory)
- `config/commands.json` (keyword matching dictionary)

## 15. 必须适配部分 (Must Adapt Parts)

- **IPC Transport**: Legacy uses `JsonlClientSender` / `JsonlInboundListener` with raw newline-delimited JSON. `embedded_com` uses **msgpack length-prefixed framing** (from `orchestrator_service.ipc.transport` / `protocol`).
- **Orchestrator Adapter**: Voice Gateway needs to import and use the correct `JsonlClientSender` and `JsonlInboundServer` from `orchestrator_service.ipc.transport` to match framing.
- **Configuration integration**: Shift from environment variables/board_config.py to unified YAML profile files.
- **Path Resolution**: Resolve relative paths to absolute paths using the workspace repo root.
- **Windows compatibility**: Run input reader from stdin/files under dry-run-text mode, disabling microphone recording (`RawMicStream`) and model loading.

## 16. 仅供参考的历史脚本 (Historical Scripts for Reference Only)

All scripts under `Voice_legacy/Voice/scripts/` will be ignored/deprecated for final integration:
- `sc171_vad_asr_kws_tts_dualthread.py`
- `sc171_vad_asr_kws_tts_dualthread_concise.py`
- `inspect_funasr_online.py`
- `test_online_model_minimal.py`
- `check_dim.py`

## 17. 缺失资源 (Missing Resources in Legacy Package)

- `thirdparty/fsmn_vad`
- `thirdparty/paraformer_asr_online`
- `thirdparty/paraformer_asr`
- `openwakeword_models`
If these models/directories do not exist in the filesystem, the Voice Gateway will raise a warning and bypass loading in `dry_run_text` mode, while halting with a structured error in normal mode.

## 18. 不应迁移的运行产物 (Do Not Migrate Runtime Artifacts)

- `runs/*`
- `logs/*`
- `pids/*`
- `tts_cache/*`
- `tts_out/*`
- `__pycache__` directories
