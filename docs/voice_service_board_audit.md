# Voice Service Board Audit

Audited package: `Voice/voice_service` in the SC171 repository. The package is
the only supported Voice product package; no external `voice_service_latest`
directory is used.

## P0 fixes

- `app/main.py` accepts `--profile`, `--inspect-config`, and
  `--list-audio-devices`; it no longer mutates `sys.path`.
- `config/__init__.py` is schema-only and no longer constructs a global config
  while importing.
- `config/loader.py` maps the profile's `audio.device` and understands phone
  feedback/playback fields. It resolves repository-relative model paths.
- Debug input mode creates only a log-only sender. It does not construct a
  TaskAck listener, local Piper, or a local TTS event listener.
- The debug profile uses the SC171 USB device
  `plughw:CARD=UACDemoV10,DEV=0`, disables TaskCmd/TaskAck/local TTS, and uses
  canonical `key` rather than `keys`.

## P1 phone-TTS work completed

`runtime/playback.py` now emits the only Voice-originated phone event
(`WAKE_PROMPT`) to the Mobile Gateway event socket and validates
`event_id`, `session_id`, epoch, expected ordering, and duplicate completion.
The state path is `WAIT_WAKE → WAIT_PROMPT_PLAYBACK → POST_PLAYBACK_GUARD →
ARMED_WAIT`. Start/finish timeouts take the finite guard path; they never claim
successful playback or retry indefinitely. STOP bumps the command epoch and
invalidates the pending playback context before a stale acknowledgement can
arm ASR.

Phone profiles disable local Piper and the legacy local `tts_event` listener.
Task acceptance/rejection/search/completion speech remains an Orchestrator
responsibility. The Mobile Gateway preserves `epoch` when converting
`tts_ack` into `tts_playback_state`, allowing Voice to reject stale receipts.

## Legacy compatibility

`board_config.py` remains a legacy fallback only. The launcher calls
`voice_service.app.main` with `--profile`, and the runtime loader reads the
selected profile. Existing framed-msgpack transport wrappers are reused rather
than inventing a parallel IPC protocol.
