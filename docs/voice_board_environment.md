# SC171 Voice board environment audit

Audit timestamp: 2026-07-11 23:15 CST.

The selected interpreter is `/usr/bin/python3` (Python 3.8.10).  It is the
only discovered candidate interpreter and is compatible with the existing
SC171 aarch64 Ubuntu 20.04 environment.  The launcher honours
`VOICE_PYTHON`, so an explicit isolated interpreter may be supplied later
without changing system packages.

| Dependency | Board result |
| --- | --- |
| numpy | 1.24.4 |
| PyYAML (`yaml`) | 5.3.1 |
| msgpack | 1.1.1 |
| onnxruntime | 1.19.2 |
| soundfile | missing |
| tflite_runtime | missing |
| openwakeword | missing |
| funasr_onnx | missing |
| piper | missing (not required for phone-TTS) |

`arecord` and `aplay` are installed (ALSA 1.2.2), but this session reports no
sound cards; `plughw:CARD=UACDemoV10,DEV=0` cannot be validated until the USB
audio device is available to the board session.  No recording or playback was
performed.  No package was installed and NumPy was not changed.

The pre-install package snapshot is
`/tmp/voice_pip_before_20260711_230740.txt`.  No matching local wheels were
found under `/home/aidlux`; installation is therefore blocked pending a
deliberate isolated-environment dependency decision, rather than falling back
to a system-wide install.
