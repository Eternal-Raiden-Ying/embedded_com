# Deterministic Voice replay audit

Before this change, `RawMicStream` was the sole runtime audio input. It starts
`arecord -D <device> -t raw -f S16_LE -r 16000 -c 1`, reads fixed 80 ms PCM
frames (`1280` samples, `2560` bytes), and passes those bytes directly to
`AudioKWSWorker`. The worker converts to `int16`, evaluates KWS, drives
`WAIT_WAKE → ARMED_WAIT → REC`, enqueues the final utterance, and the existing
`ASRDecisionWorker` runs VAD, offline Paraformer, intent/target parsing, and
TaskCmd/TaskAck through the existing framed-msgpack adapter.

`WavReplayAudioSource` reuses that exact `read_frame()/close()/stats()`
boundary. It validates 16 kHz mono S16_LE WAV input, produces identical frame
sizes, schedules frames from a monotonic clock when realtime is enabled, and
does not duplicate KWS, segmentation, ASR, intent, or IPC logic.

Existing reusable pieces are `RawMicStream`, `AudioKWSWorker`,
`AudioCommandPipeline`, `dispatch_task_cmd`, `JsonlAckInbox`, and the
Orchestrator UDS TaskCmd/TaskAck endpoints. The previous `app/main.py`
audio-replay helper is an offline probe, not a runtime microphone path, and is
therefore retained for diagnostics rather than duplicated. No production
arecord path, protocol, safety gate, or existing run artifact was removed.
