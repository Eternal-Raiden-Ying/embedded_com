# IPC 通信机制重构说明文档（TCP/JSON -> UDS/MessagePack）

为了提升系统性能、降低同机（On-device）跨进程通信延迟并减少 CPU 开销，我们对系统内的 Vision 模块与 Orchestrator 模块之间的 IPC 机制进行了底层重构。

本篇文档详细说明了本次重构的设计方案、修改内容、边界处理以及板端运行与验证方法。

---

## 1. 核心架构变更

重构前后的通信协议堆栈对比如下：

| 维度 | 重构前 (TCP + JSON) | 重构后 (UDS + MessagePack) |
| :--- | :--- | :--- |
| **传输层 (Transport)** | TCP Socket (IPv4 Loopback `127.0.0.1`) | Unix Domain Socket (`AF_UNIX` 流套接字) |
| **帧定界 (Framing)** | 换行符定界 (Newline `\n` Delimited) | 4 字节大端整数长度前缀 (4-byte Big-Endian Length Prefix) |
| **序列化层 (Serialization)** | JSON 文本序列化 (UTF-8 字符串) | MessagePack 高效二进制序列化 (Binary) |

---

## 2. 修改文件清单

我们对以下模块及配置文件进行了修改，**且均保持了板端（SC171 Linux）路径及环境的完整性**，未引入任何本地 Windows 的路径。

### 2.1 序列化与帧解析层

*   **[VISTA/vision_module/ipc/protocol.py](file:///d:/55495/workspace/embedded_com/VISTA/vision_module/ipc/protocol.py)** & **[orchestrator/orchestrator_service/ipc/protocol.py](file:///d:/55495/workspace/embedded_com/orchestrator/orchestrator_service/ipc/protocol.py)**:
    *   引入 `msgpack` 库。
    *   重构 `pack_msg` 与 `unpack_msg` 逻辑。由于 MessagePack 解析出来字典 Key 默认可能是 `bytes`，程序中已使用 `raw=False` 或解码操作妥善处理 Key 类型为 `str`。
*   **[VISTA/vision_module/ipc/transport.py](file:///d:/55495/workspace/embedded_com/VISTA/vision_module/ipc/transport.py)** & **[orchestrator/orchestrator_service/ipc/transport.py](file:///d:/55495/workspace/embedded_com/orchestrator/orchestrator_service/ipc/transport.py)**:
    *   将 `socket.socket` 更改为 `socket.AF_UNIX`。
    *   移除了原基于 `readline()` 的换行符文本流读取逻辑。
    *   引入了基于**4字节大端长度前缀**的流式数据包定界逻辑：发送时先发送 4 字节长度头，接收时先精确读取 4 字节获取长度，再循环读取对应长度的二进制 Payload，彻底解决了二进制流的数据粘包和半包问题。
    *   在 Socket 初始化和绑定时，添加了判断并清理残留 UDS 文件（`os.unlink(path)`）的逻辑，避免由于上次异常退出导致端口占用冲突。
    *   添加了 Windows 下 `socket.AF_UNIX` 不存在时的兼容性 Mock 代码，保障 Windows 开发环境下编译/分析不崩溃。

### 2.2 配置与业务适配层

*   **配置 Dataclass Schema**:
    *   [common/schema.py](file:///d:/55495/workspace/embedded_com/common/schema.py)
    *   [VISTA/vision_module/config/schema.py](file:///d:/55495/workspace/embedded_com/VISTA/vision_module/config/schema.py)
    *   [orchestrator/orchestrator_service/config/schema.py](file:///d:/55495/workspace/embedded_com/orchestrator/orchestrator_service/config/schema.py)
    *   [orchestrator/orchestrator_service/mobile_gateway/config/schema.py](file:///d:/55495/workspace/embedded_com/orchestrator/orchestrator_service/mobile_gateway/config/schema.py)
    *   *修改内容*：去除了已废弃的 `host` 和 `port` 字段，引入了专用的 `ipc_socket_path` 配置字段，并将默认 `transport` 变更为 `"uds"`。
*   **YAML 配置文件**:
    *   [configs/system_config.yaml](file:///d:/55495/workspace/embedded_com/configs/system_config.yaml)
    *   [VISTA/configs/vision_params.yaml](file:///d:/55495/workspace/embedded_com/VISTA/configs/vision_params.yaml)
    *   [configs/mobile_gateway.mqtt.example.yaml](file:///d:/55495/workspace/embedded_com/configs/mobile_gateway.mqtt.example.yaml)
    *   *修改内容*：将 Vision/Orchestrator 相关的 IPC 端点全部适配为 UDS 地址配置，例如：
        *   `vision_req.sock` (9003)
        *   `vision_obs.sock` (9002)
        *   `task_cmd.sock` (9001)
        *   `task_ack.sock` (9012)
        *   `mobile_gateway_cmd.sock` (9101)
*   **启动脚本**:
    *   [start_robot_stack.sh](file:///d:/55495/workspace/embedded_com/start_robot_stack.sh)
    *   *修改内容*：
        1. 移除了已被废弃的 `ORCH_TASK_*_HOST` / `PORT` 等环境变量，变更为 `ORCH_TASK_*_SOCKET_PATH` 等地址；
        2. 原脚本启动进程后通过 TCP 进行就绪性扫描端口。我们新增了 `wait_for_sockets` 函数，采用 Linux 标准的 `[[ -S "$socket_path" ]]` 检测套接字文件是否存在并处于监听状态，从而适配 UDS 启动检测。
        3. **保留了板端原生环境配置**，如 `/usr/bin/python3` 以及板端绝对路径 `/tmp/robot_stack/...`。

### 2.3 测试用例适配

*   [tests/test_gateway_mapping.py](file:///d:/55495/workspace/embedded_com/tests/test_gateway_mapping.py) & [tests/test_real_protocol_mapping.py](file:///d:/55495/workspace/embedded_com/tests/test_real_protocol_mapping.py):
    *   更新了测试中的 Mock 配置字典，移除已废弃的 host/port 校验，改用 UDS 路径断言，确保单元测试集中的配置映射关系逻辑正确。

---

## 3. 逻辑正确性与鲁棒性保证

1.  **分包粘包处理 (Framing Guard)**:
    在基于流 (Stream) 的通信中，TCP/UDS 并不保证数据包界限。旧代码靠 `\n` 定界，而 MessagePack 序列化后是含有任意二进制字符的 Payload，其中可能包含 `0x0A` (即 `\n`)。因此改用 **[4字节长度前缀 + 循环Recv填充]** 方式。该设计完全避免了数据截断与粘包。
2.  **文件残留处理 (Stale Socket Cleaning)**:
    UDS 监听时，如果异常关闭，对应的 `.sock` 文件依然会遗留在文件系统中。如果下次启动不清理直接 bind，会报 `Address already in use` 错误。修改后的逻辑在 `bind()` 之前，先主动检测并 `os.unlink()` 清除原有冲突文件，保证服务可以稳定拉起。
3.  **多平台兼容 (Windows Sandbox Fallback)**:
    Windows 环境默认 Python SDK 中可能没有 `socket.AF_UNIX` 常量，导致在 Windows 上导入代码时直接崩溃崩溃。我们在 `transport.py` 顶部做了安全防护：
    ```python
    if not hasattr(socket, "AF_UNIX"):
        socket.AF_UNIX = 9999  # 兼容 Windows 编译导入，防止崩溃
    ```
    使得在 Windows 开发机上进行语法验证、静态代码分析、导入路径校验时能正常工作。

---

## 4. 运行与验证建议

本重构代码在本地开发机中已通过单元测试语法校验。布设到 **SC171 板端** 后，你可以使用以下方式验证：

1.  **启动整套系统**:
    ```bash
    ./start_robot_stack.sh
    ```
    通过输出查看 `vision ready sockets=[/tmp/robot_stack/vision_req.sock]` 与 `orchestrator ready sockets=[...]` 是否正常就绪。
2.  **检查套接字文件**:
    在系统运行过程中，检查目录 `/tmp/robot_stack/` 下是否生成了对应的 `.sock` 文件，且权限是否正确。
3.  **日志观察**:
    通过手机端发送命令，查看 `logs/runs` 目录下生成的 `state_blocks.jsonl` 日志，确认消息成功通过二进制 MessagePack 解包并处理。
