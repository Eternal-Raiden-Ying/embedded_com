# Voice Gateway Model Inventory Audit

This document records the audit of the speech model assets present in the `embedded_com/Voice` directory. All sizes and file details represent the physical status verified directly on the filesystem.

---

## 1. ASR & VAD Models (`Voice/ONNX/`)

### 1.1 VAD Model Directory: `speech_fsmn_vad_zh-cn-16k-common-onnx`
* **Component**: `vad`
* **Relative Path**: `Voice/ONNX/speech_fsmn_vad_zh-cn-16k-common-onnx`
* **Type**: `directory`
* **Required By**: Offline ASR Pipeline (FSMN-VAD segment cutter)
* **Expected Backend**: `funasr_onnx` (FSMN-VAD)
* **Tracked by Git**: `No` (Ignored under `ONNX/`)
* **Ignored by Git**: `Yes`
* **Current Status**: `FOUND` (Complete companion files)
* **Notes**: Contains the ONNX VAD graph, model configuration, and mean-variance normalization (MVN) file.

#### Files inside VAD directory:
| Filename | Type | Size (Bytes) | Ignored | Notes |
| --- | --- | --- | --- | --- |
| `model_quant.onnx` | `onnx` | 506,744 | Yes | The quantized VAD graph. |
| `am.mvn` | `mvn` | 8,040 | Yes | Acoustic model mean-variance normalization. |
| `config.yaml` | `yaml` | 1,271 | Yes | Runtime config parameters for FSMN-VAD. |
| `configuration.json` | `json` | 619 | Yes | Metadata configuration. |
| `README.md` | `txt` | 5,067 | Yes | Documentation. |
| `quickstart.md` | `txt` | 310 | Yes | Documentation. |

### 1.2 ASR Model Directory: `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx`
* **Component**: `asr`
* **Relative Path**: `Voice/ONNX/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx`
* **Type**: `directory`
* **Required By**: ASR Transcriber (Offline/Online Paraformer)
* **Expected Backend**: `funasr_onnx` (Paraformer)
* **Tracked by Git**: `No` (Ignored under `ONNX/`)
* **Ignored by Git**: `Yes`
* **Current Status**: `FOUND` (Complete companion files)
* **Notes**: Contains the main Paraformer weights, token vocabulary dictionary, and MVN metadata.

#### Files inside ASR directory:
| Filename | Type | Size (Bytes) | Ignored | Notes |
| --- | --- | --- | --- | --- |
| `model_quant.onnx` | `onnx` | 238,380,216 | Yes | Quantized Paraformer ASR network. |
| `am.mvn` | `mvn` | 11,211 | Yes | Acoustic model mean-variance normalization. |
| `config.yaml` | `yaml` | 2,632 | Yes | ASR network options. |
| `configuration.json` | `json` | 652 | Yes | Metadata configuration. |
| `tokens.json` | `json` | 102,081 | Yes | Word token vocab lookup table (8404 entries). |
| `README.md` | `txt` | 5,776 | Yes | Documentation. |
| `quickstart.md` | `txt` | 312 | Yes | Documentation. |

---

## 2. KWS Models (`Voice/kws/`)

* **Component**: `wake_kws` / `stop_kws`
* **Relative Path**: `Voice/kws/`
* **Tracked by Git**: `No` (Only the directory `README.md` is tracked)
* **Ignored by Git**: `Yes` (ONNX binaries under KWS are ignored)
* **Expected Backend**: `openwakeword` (ONNX classifier)
* **Notes**: Uses openwakeword's built-in feature extractor (melspectrogram + embedding networks) to pass features into these custom wake classifiers.

#### Files in KWS:
| Filename | Component | Type | Size (Bytes) | Ignored | Notes |
| --- | --- | --- | --- | --- | --- |
| `wake_nihao_xiaoche_v2.onnx` | `wake_kws` | `onnx` | 415,224 | Yes | Wake word classifier ("你好小车"). |
| `wake_nihao_xiaoche_clean.onnx` | `wake_kws` | `onnx` | 415,224 | Yes | Alternative wake word classifier. |
| `stop_smallcar_v1.onnx` | `stop_kws` | `onnx` | 415,224 | Yes | Stop hotword classifier ("停" / "小车停下"). |
| `README.md` | `n/a` | `txt` | 364 | No | Directory explanation. |

---

## 3. TTS Models (`Voice/tts/`)

* **Component**: `tts`
* **Relative Path**: `Voice/tts/`
* **Tracked by Git**: `No` (Only the directory `README.md` is tracked)
* **Ignored by Git**: `Yes` (ONNX and JSON files under TTS are ignored)
* **Expected Backend**: `piper` (Python Piper or Piper CLI)
* **Notes**: Model operates at 16000Hz sample rate. Requires paired ONNX and JSON files.

#### Files in TTS:
| Filename | Component | Type | Size (Bytes) | Ignored | Notes |
| --- | --- | --- | --- | --- | --- |
| `zh_CN-huayan-x_low/zh_CN-huayan-x_low.onnx` | `tts` | `onnx` | 20,628,813 | Yes | Piper TTS synthesis network. |
| `zh_CN-huayan-x_low/zh_CN-huayan-x_low.onnx.json` | `tts` | `json` | 4,164 | Yes | Voice parameters, speaker list, and phoneme dict. |
| `README.md` | `n/a` | `txt` | 176 | No | Directory explanation. |
