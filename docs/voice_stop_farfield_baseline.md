# STOP-FARFIELD-001 — Deferred

The known near-field sample `Voice/runs/manual_audio/stop_01.wav` has a Stop
KWS score of approximately **0.991**. It remains the only current Stop sample
used for replay acceptance.

The first far-field batch (`farfield_stop_repeat_20260712_114256`) is
transcribed by offline ASR as “小车停止”, but its Stop KWS scores are generally
below the 0.580 threshold. The second, closer batch was re-parsed from
`kws_results.log` with the repository audit command below rather than copied
by hand:

```bash
awk '/^===== / {sample=$2} /Stop Max Score/ {print sample, $NF} /Stop Triggered/ {print sample, $NF}' \
  Voice/runs/farfield_stop_repeat_20260712_114452/kws_results.log
```

| Sample | Stop score | Triggered |
| --- | ---: | --- |
| 01 | 0.001 | false |
| 02 | 0.993 | true |
| 03 | 0.855 | true |
| 04 | 0.040 | false |
| 05 | 0.670 | true |

Conclusion: this is a model far-field robustness and pronunciation-sensitivity
limitation, not an IPC or state-machine failure. The status is
**STOP-FARFIELD-001 / Deferred**. `stop_th` is not reduced. This limitation
does not block the deterministic FIND replay integration, and far-field Stop
must not be treated as the only current safety-stop mechanism.
