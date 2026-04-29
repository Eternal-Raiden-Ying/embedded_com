# Mobile Control Repo Analysis

此文档保留为历史分析记录。当前真实运行链路以 [system_runbook.md](/home/aidlux/embedded_com/docs/system_runbook.md) 为准。

## 当前结论

- 手机小程序/云端 MQTT 是任务入口。
- `mobile_gateway` 订阅 `robot/v1/SC171/mobile/cmd`，向 Orchestrator 发送 `task_cmd`。
- Orchestrator 继续负责状态机、VISTA 请求和 STM32 控制。
- VISTA 继续通过 `vision_req` / `vision_obs` 与 Orchestrator 通信。
- 板端 `Voice/ASR` 服务已从仓库归档，不再作为运行组件。

## 当前链路

```text
小程序/云 MQTT -> mobile_gateway -> Orchestrator -> VISTA -> STM32
```

端口：

| 链路 | 默认地址 |
|------|----------|
| `task_cmd` | `127.0.0.1:9001` |
| `vision_obs` | `127.0.0.1:9002` |
| `vision_req` | `127.0.0.1:9003` |
| `task_ack` | `127.0.0.1:9012` |

`tts_event` 仅保留兼容字段，默认禁用。
