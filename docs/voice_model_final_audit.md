# Voice Model Final Integration & Audit Report

This report records the final audit status, KWS backend configurations, target capability gating, pytest validation results, and git status of the Voice Gateway in the `embedded_com` repository.

---

## 1. 模型目录树 (Model Directory Tree)

No redundant `Voice/ONNX/onnx` nesting level was created. The layout of the model assets is as follows:

```text
Voice/
├── ONNX/
│   ├── speech_fsmn_vad_zh-cn-16k-common-onnx/
│   │   ├── am.mvn (8,040 bytes)
│   │   ├── config.yaml (1,271 bytes)
│   │   ├── configuration.json (619 bytes)
│   │   └── model_quant.onnx (506,744 bytes)
│   └── speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/
│       ├── am.mvn (11,211 bytes)
│       ├── config.yaml (2,632 bytes)
│       ├── configuration.json (652 bytes)
│       ├── model_quant.onnx (238,380,216 bytes)
│       └── tokens.json (102,081 bytes)
├── kws/
│   ├── stop_smallcar_v1.onnx (415,224 bytes)
│   ├── wake_nihao_xiaoche_clean.onnx (415,224 bytes)
│   └── wake_nihao_xiaoche_v2.onnx (415,224 bytes)
└── tts/
    └── zh_CN-huayan-x_low/
        ├── zh_CN-huayan-x_low.onnx (20,628,813 bytes)
        └── zh_CN-huayan-x_low.onnx.json (4,164 bytes)
```

---

## 2. Model Manifest Resolved Paths & Hashes

The model paths resolved from `model_manifest.yaml` (verified using `inspect_models.py`):

| Model Component | Configured Path | Resolved Path | SHA256 Hash |
| :--- | :--- | :--- | :--- |
| **ASR Model** | `Voice/ONNX/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx` | `D:\55495\workspace\embedded_com\Voice\ONNX\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx` | `c3a06538f867c8e3d9bb9c9e1b44b126a50d732862f6c428b1b5896ee9be65c5` (`model_quant.onnx`) |
| **VAD Model** | `Voice/ONNX/speech_fsmn_vad_zh-cn-16k-common-onnx` | `D:\55495\workspace\embedded_com\Voice\ONNX\speech_fsmn_vad_zh-cn-16k-common-onnx` | `52895ab1bc270529d20c57173de71eb058728d7a1262d1c6fb6ea2ef82b99a61` (`model_quant.onnx`) |
| **Wake KWS** | `Voice/kws/wake_nihao_xiaoche_v2.onnx` | `D:\55495\workspace\embedded_com\Voice\kws\wake_nihao_xiaoche_v2.onnx` | `2bbc079afb5436c361f528a70d117b4876535465f40f8db93a1a43d83a95d6eb` |
| **Stop KWS** | `Voice/kws/stop_smallcar_v1.onnx` | `D:\55495\workspace\embedded_com\Voice\kws\stop_smallcar_v1.onnx` | `27fc24b040bc26934f9e8ca995d564230d7e85d69ed6f275b93a4b602cd0b3f3` |
| **TTS Model** | `Voice/tts/zh_CN-huayan-x_low/zh_CN-huayan-x_low.onnx` | `D:\55495\workspace\embedded_com\Voice\tts\zh_CN-huayan-x_low\zh_CN-huayan-x_low.onnx` | `d30b143fac66d821a1285aa013295adf5cd129d3cc11d70334e51c7b20662c37` |
| **TTS Config** | `Voice/tts/zh_CN-huayan-x_low/zh_CN-huayan-x_low.onnx.json` | `D:\55495\workspace\embedded_com\Voice\tts\zh_CN-huayan-x_low\zh_CN-huayan-x_low.onnx.json` | `5521dcb09adf68a9bee289032f7f5af18d29bff020953429b5d223ec1f881816` |

> [!NOTE]
> All model hashes were calculated and verified before and after the audit, remaining completely unchanged (read-only assets).

---

## 3. KWS Frontend & Classifier Backends

KWS configuration settings are explicitly declared and resolved:
- **SC171 Production Profile (`sc171_voice_gateway.yaml` / `sc171_hybrid.yaml`)**:
  - `frontend_backend: tflite` (matching original model training frontend)
  - `classifier_backend: onnx`
- **Windows Dev Profile (`windows_voice_dev.yaml`)**:
  - `frontend_backend: onnx`
  - `classifier_backend: onnx`

---

## 4. Windows KWS/ASR/TTS Actual Status

- **Inspect Models Status**:
  - **All components** (`wake_kws`, `stop_kws`, `vad`, `asr`, `tts`, `tts_config`) successfully mapped to `PASS` in static analysis mode, with correct input/output shape probing under `CPUExecutionProvider`.
- **Inference Probe Status**:
  - Since Windows lacks the raw `tflite-runtime` library and `melspectrogram.onnx` package resources to compute OpenWakeWord mel-spectrogram embeddings locally, running KWS in real mode outputs `SKIPPED_ENV_DEPENDENCY` or `FAIL` gracefully.
  - Windows verification uses dry-run mode (`dry_run_text: true`) successfully to execute mock/stub inferences.

---

## 5. Target Catalog Testing Results

- **Authoritative Gate Gating**:
  - `configs/target_catalog.yaml` defines `key` as `executable` and `remote`, `medicine_box`, `eye_drops`, `nail_clipper`, `battery` as `model_pending`.
  - Pytest case `test_target_catalog_rejection` successfully verifies that if the target is `remote`, `handle_task_cmd` in `TaskRuntimeMixin` rejects the request returning `accepted=False` and `reason="target_recognized_but_not_executable"`.
  - Emits the correct warning and queues the designated TTS: `该物品模型尚在开发中，无法获取` without entering `SEARCH_TABLE` or creating any vision request.

---

## 6. Pytest Complete Results

```text
tests/voice/test_voice_gateway.py::test_path_resolution PASSED           [ 11%]
tests/voice/test_voice_gateway.py::test_intent_matching PASSED           [ 22%]
tests/voice/test_voice_gateway.py::test_stop_state_updates PASSED        [ 33%]
tests/voice/test_voice_gateway.py::test_model_missing_checks PASSED      [ 44%]
tests/voice/test_voice_gateway.py::test_ipc_protocol_validation PASSED   [ 55%]
tests/voice/test_voice_gateway.py::test_tcp_ipc_roundtrips PASSED        [ 66%]
tests/voice/test_voice_gateway.py::test_stop_protection PASSED           [ 77%]
tests/voice/test_voice_gateway.py::test_ack_routing PASSED               [ 88%]
tests/voice/test_voice_gateway.py::test_target_catalog_rejection PASSED  [100%]

============================== 9 passed in 1.13s ==============================
```

---

## 7. Shell Syntax Check Results

All shell scripts were syntactically checked.
- No syntax warnings or syntax errors were found in any script.
- Verified files:
  - `Voice/scripts/board_preflight.sh`
  - `Voice/scripts/board_acceptance.sh`
  - `Voice/start_voice_asr.sh`
  - `start_robot_stack.sh`

---

## 8. Board Acceptance No Motion Guarantee

Probes and acceptance script default loops are side-effect free:
- **`kws_probe`**: Default does not send TaskCmd.
- **`asr_probe`**: Default does not send TaskCmd.
- **`tts_probe`**: Default only saves output WAV (playback disabled).
- **`audio_device_probe`**: Default lists card hardware without recording/playing.
- **`audio-replay`**: Default maps `--no-task-send` as active.
- **`board_acceptance.sh`**: Does not contain any command driving bottom chassis motors or real UART serial pathways.

---

## 9. Git Diff Stat

```text
 .../orchestrator_service/runtime/task_runtime.py   |  26 ++++
 start_robot_stack.sh                               | 152 ++++++++++++++++++---
 2 files changed, 160 insertions(+), 18 deletions(-)
```

---

## 10. Git Status

```text
On branch feature/docking-control-v3
Changes not staged for commit:
	modified:   orchestrator/orchestrator_service/runtime/task_runtime.py
	modified:   start_robot_stack.sh

Untracked files:
	Voice/
	configs/profiles/sc171_hybrid.yaml
	configs/profiles/sc171_voice_gateway.yaml
	configs/profiles/windows_voice_dev.yaml
	configs/target_catalog.yaml
	docs/voice_legacy_audit.md
	docs/voice_model_final_audit.md
	docs/voice_model_inventory.md
	tests/voice/
	walkthrough.md
```

---

## 11. 尚需 SC171 验证的内容 (Remaining Board Verifications)

1. **Production KWS Verification**: Test the `tflite` melspectrogram/embedding features on the SC171 ARM processor using `kws_probe.py --profile configs/profiles/sc171_voice_gateway.yaml`.
2. **Offline ASR RTF Baseline**: Run `asr_probe.py --mode offline` on the board to establish the Paraformer large model's Real-Time Factor.
3. **Sound Capture Hardware Link**: Test the microphone card array input capture stability under real ALSA `arecord` processes.
4. **Local TTS Speaker Output**: Synthesize and output the TTS prompt through `aplay` to check volume and audio clarity.
