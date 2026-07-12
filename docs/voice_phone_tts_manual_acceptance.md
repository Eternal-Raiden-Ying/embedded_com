# Voice and phone-TTS manual acceptance

All commands run from `/home/aidlux/embedded_com`. Do not start UART, chassis,
arm, VISTA, or a full robot task during Gates A–G.

## Gate A — import and profile configuration

Prerequisite: services stopped. Run:

```bash
export REPO_ROOT="$PWD"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/Voice:$REPO_ROOT/common:$REPO_ROOT/orchestrator"
python3 -c 'import voice_service; print(voice_service.__file__)'
python3 -m voice_service.app.main --profile configs/profiles/sc171_voice_input_debug.yaml --inspect-config
```

Pass when the package is under `Voice/voice_service`, the device is
`plughw:CARD=UACDemoV10,DEV=0`, and all task/ack/mobile/Piper endpoints are
disabled. On failure check `Voice/voice_service/config/loader.py` and the
selected profile.

## Gate B — model inspection

Prerequisite: Gate A. Run:

```bash
python3 -m voice_service.examples.inspect_models --profile configs/profiles/sc171_voice_input_debug.yaml
```

Pass when wake, STOP, ASR, and VAD paths exist below `Voice/kws` or
`Voice/ONNX`; Piper must show `DISABLED_OPTIONAL`. On failure inspect
`Voice/config/model_manifest.yaml`. No audio is played.

## Gate C/D — recorded-file KWS and ASR probes

Prerequisite: a user-supplied 16 kHz mono PCM WAV, for example
`/path/to/voice.wav`. Run:

```bash
python3 -m voice_service.examples.kws_probe --profile configs/profiles/sc171_voice_input_debug.yaml --wav /path/to/voice.wav
python3 -m voice_service.examples.asr_probe --profile configs/profiles/sc171_voice_input_debug.yaml --wav /path/to/voice.wav --mode offline
```

The user supplies a recording containing “你好小车” or “帮我拿苹果”. Pass when
the probes print backend, score/text, intent, and canonical target. They must
not send TaskCmd (do not pass `--send-task`). If dependencies are missing,
check the selected Python environment rather than replacing models or adding
mocks.

## Gate E — live safe input-debug

Prerequisite: USB audio appears in `arecord -l`; no robot components running.
First run the read-only permission preflight:

```bash
bash Voice/scripts/check_usb_mic_access.sh
```

The SC171 Android audio node uses GID `1005`, while the usual AidLux audio GID
is `29`. ACLs are unsupported on `/dev`; ask an administrator to add `aidlux`
to the existing GID-1005 group, then re-login SSH/VS Code. `chgrp audio` is a
temporary workaround only and may be lost after reboot or USB replug. The
Voice launcher never runs sudo.

Run exactly:

```bash
cd /home/aidlux/embedded_com

export REPO_ROOT="$PWD"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/Voice:$REPO_ROOT/common:$REPO_ROOT/orchestrator"

bash Voice/start_voice_asr.sh fg \
  --profile configs/profiles/sc171_voice_input_debug.yaml
```

Say “你好小车”, “帮我拿苹果”, and “小车停止”. Expected logs include
`[VOICE][READY]`, `[VOICE][WAKE]`, state transitions, VAD/ASR/INTENT, and
`[VOICE][TASK] suppressed`; no TaskCmd socket, MQTT, Piper, playback socket,
or robot action is permitted. A failure here belongs to the audio device/KWS/
ASR environment, not the robot stack.

## Gate F — mini-program TTS transport

Prerequisite: Gate E plus an intentionally started Mobile Gateway and its
configured MQTT broker; this is a separate, supervised test. Use the
`sc171_voice_phone_tts_dryrun.yaml` profile, keep serial and arm serial dry
run, and do not start VISTA or any hardware motion. Trigger a Voice wake.
The mini-program should receive a `tts_event` with phrase `WAKE_PROMPT`, text
“我在，请说出要拿的物品。”, plus event/session/epoch identifiers. It returns
`started` then `finished` echoing those identifiers. Check Mobile Gateway MQTT
and `tts_playback_state` logs if it fails.

## Gate G — playback guard behavior

Prerequisite: Gate F. During phone playback, say ordinary words: no ASR
recording or new wake session may start; STOP KWS remains usable. After
`finished`, wait 350 ms and then speak the command. Pass when Voice moves from
`WAIT_PROMPT_PLAYBACK` through `POST_PLAYBACK_GUARD` to `ARMED_WAIT`; stale,
wrong, and duplicate acknowledgements must not arm it. Do not start UART,
chassis, arm, VISTA, or a complete task.
