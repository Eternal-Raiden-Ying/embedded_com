# Stack launcher Voice / Vision dry-run audit

## Scope

The launcher now accepts exactly one task-input producer: `mobile` or `voice`.
`FEEDBACK_OUTPUT_MODE` is separate (`disabled` or retained future `phone_tts`);
this dry-run integration uses `disabled` only.  `hybrid` is rejected so two
TaskCmd producers cannot race.
`ROBOT_INPUT_MODE` is launcher-only and cannot override Voice's audio source;
profiles (or explicit `VOICE_INPUT_MODE`) select `arecord` versus `wav_replay`.

## Confirmed historical regression

At `b103883`, an invocation with no arguments took `main -> action=start ->
apply_profile_defaults -> start_stack`; it was not a Gateway-only command.
For mobile mode it started the Gateway, then unconditionally called
`wait_for_endpoint mobile_gateway gateway_tts_event_in`.  In `dry_run` that
endpoint is disabled.  The old checker waited for a PID and a READY log even
though no socket could exist, timed out, and exited before `start_core`.
The Gateway therefore looked healthy while VISTA and Orchestrator had not
started.

`git diff 18ce0fe4cdda43bca383824147a57c9903db7ee1..HEAD --
start_robot_stack.sh` shows the later TTS endpoint wait and Voice/hybrid
additions.  The historical version started Gateway then Core without a TTS
socket readiness dependency.  This change restores that separation without
removing the future TTS route.

## Endpoint ownership and ordering

| Socket | Listener owner | Connector |
| --- | --- | --- |
| `task_cmd.sock` | Orchestrator | Mobile Gateway or Voice (one only) |
| `task_ack.sock` | Voice in Voice mode | Orchestrator |
| `vision_req.sock` | VISTA | Orchestrator |
| `vision_obs.sock` | Orchestrator | VISTA |

Mobile: VISTA, Orchestrator, Mobile Gateway.  Live Voice: VISTA, Voice
(TaskAck listener), Orchestrator.  The finite `wav_replay` source is the one
necessary exception: it consumes its manifest as soon as model loading ends,
so it runs VISTA, Orchestrator, Voice to ensure a real TaskCmd cannot precede
the listener.  All connectors retain their existing retry behavior.
TTS endpoints are optional in this release: disabled optional endpoints print
`[SKIP]`; a disabled required task/vision endpoint fails immediately.

## Logs and safety

An integrated run shares `STACK_RUN_ID` and writes to
`logs/runs/<run_id>/{launcher,gateway,voice,vision,orchestrator,summary}`.
`launcher/manifest.json` records profile, endpoint map, dry-run flags and PID
file paths.  The dedicated Voice profile uses real `arecord` input but sets
serial and arm dry-run, TTS/mobile feedback/playback disabled, offline ONNX
KWS and quantized offline ASR/VAD.  It does not request text injection.
