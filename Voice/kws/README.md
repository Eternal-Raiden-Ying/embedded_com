# KWS Models Directory

The production Voice Gateway uses one Sherpa-ONNX streaming keyword spotter for
both wake and STOP phrases. It consumes frames from the Gateway's existing
audio capture worker; it does not open an audio device itself.

## Production resources

- `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/`
- `sherpa_custom/extreme.txt`

Canonical paths and runtime parameters live in
`configs/common/voice_gateway.yaml`. Legacy standalone classifier models may
remain in this directory for reference, but production profiles do not load
them.
