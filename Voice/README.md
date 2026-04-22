# Voice

语音模块。这个目录同时包含主服务代码 `voice_service/`，以及运行时依赖的模型、第三方 ASR/VAD 资源和保留下来的历史脚本。

## 当前定位

- `voice_service/`：现行主入口，`python3 -m voice_service.app.main`
- `kws/`、`tts/`、`thirdparty/`、`config/`：主服务默认读取的资源目录
- `scripts/`：历史原型脚本，保留作参考，不再作为主运行入口

## 目录结构

```text
Voice/
|- voice_service/
|  |- app/
|  |- config/
|  |- ipc/
|  |- runtime/
|  `- examples/
|- config/
|  |- commands.json
|  `- commands.example.json
|- kws/
|- tts/
|- thirdparty/
|  |- fsmn_vad/
|  |- paraformer_asr_online/
|  |- paraformer_asr/
|  `- openwakeword_models/
|- scripts/
|- start_voice_asr.sh
|- inspect_funasr_online.py
`- ONLINE_ASR_MIGRATION_NOTES.md
```

## 运行

```bash
# 通过全栈脚本启动（推荐）
./start_robot_stack.sh

# 直接启动语音模块
bash Voice/start_voice_asr.sh restart

# 手动前台启动（需 conda env: asr）
conda activate asr
cd /home/aidlux/embedded_com/Voice
python3 -m voice_service.app.main
```

## 配置约定

主服务默认以 `Voice/` 自身作为 `VOICE_SERVICE_ROOT` 与 `VOICE_ROOT`。

也就是说，默认会从本模块目录下读取：

- `kws/`：唤醒词与停止词模型
- `tts/`：Piper TTS 模型
- `thirdparty/fsmn_vad/`：VAD 资源
- `thirdparty/paraformer_asr_online/`：在线 ASR 资源
- `config/commands.json`：默认命令映射

## Git 管理

- `Voice/voice_service/` 走普通 Git，保存主服务源码与配置
- `Voice/` 中的大模型与必要二进制通过 Git LFS 管理
- `logs/`、`runs/`、`pids/`、`tts_cache/`、`tts_out/` 等运行期产物不纳入版本控制
