# Online ASR 低风险过渡版说明

这版修改做的是方案 A：

- 保留 KWS、energy 分段、状态机大框架
- 离线模式仍然保留，可回退
- 新增 `asr_mode=online` 的在线 ASR 路径
- 在 `REC` 期间按 chunk 推送到 ASR 线程
- `REC_END` 时发送 `FINAL`，在 `is_final=True` 下拿最终文本
- CommandInterpreter / TTS / IPC 逻辑保持不变

## 新增环境变量

- `VOICE_ASR_MODE=online|offline`
- `VOICE_ASR_ONLINE_CHUNK_FRAMES=6`
- `VOICE_ASR_ONLINE_CHUNK_SIZE=0,8,4`
- `VOICE_ASR_ONLINE_ENCODER_LOOK_BACK=4`
- `VOICE_ASR_ONLINE_DECODER_LOOK_BACK=1`
- `VOICE_ASR_EMIT_PARTIAL=1`

## 默认行为

当前默认 `VOICE_ASR_MODE=online`。
如需快速回退：

```bash
export VOICE_ASR_MODE=offline
```

## 说明

由于当前容器里没有安装 `funasr_onnx`，这版代码只做了语法级检查，
没有在真实板端模型上完成运行验证。
如果你的板端 `funasr_onnx` Online 接口参数名和本代码不同，
需要根据实际报错再微调 `runtime/asr_engine.py` 里 `OnlineASREngine._build_backend()`
和 `_call_backend()` 的候选调用方式。
