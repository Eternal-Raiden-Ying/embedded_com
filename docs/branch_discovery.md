# VISTA IPC 架构发现与建议

分支日期：2026-05-02 ~ 2026-05-04
来源：架构审计分支 — 梳理 VISTA 订阅/生产消息格式、消费路径、状态变更

---

## 一、Grasp Remote 接入前必须修改

### 1.1 Orchestrator 侧：`make_grasp_req()` + target→class_id 映射

**涉及文件**:
- `orchestrator/orchestrator_service/ipc/protocol.py` — 新增函数
- `orchestrator/orchestrator_service/utils/target_utils.py` — 新建

**关键代码** (protocol.py:622-649，现有 `make_vision_req` 参考):

```python
def make_vision_req(target=None, session_id="", epoch=0, req_id="",
                    *, op="START", stage="SEARCH", mode_hint="",
                    payload=None) -> Dict[str, Any]:
    return VisionReqMsg(
        ts=now_ts(), op=_upper_text(op, "START"),
        stage=_upper_text(stage, "SEARCH"),
        target=target, mode_hint=mode_hint,
        session_id=session_id, req_id=req_id or _new_id("req"),
        epoch=int(epoch), payload=payload,
    ).to_dict()
```

**改动**: 新增 `make_grasp_req()` — 专门处理 `stage=GRASP, mode_hint=GRASP_REMOTE, payload.class_id` 的组合。

### 1.2 VISTA 侧：`GraspStagePlan` 加 `status=="success"` 检查

**涉及文件**: `VISTA/vision_module/app/stages/grasp.py`

**关键代码** (grasp.py:425-441):

```python
if matched and last_action == "PREDICT" \
   and bool(remote_result.get("last_ok", False)) \
   and bool(remote_result.get("has_result", False)):
    if stage_state.get("remote_result_sent", False):
        return None
    stage_state["remote_result_sent"] = True
    result = deepcopy(remote_result.get("result") or {})
    # ← 此处缺少对 result["status"] == "success" 的判断
    result.setdefault("target", ctx.target_name)
    result["source"] = "remote_grasp_client"
```

**改动**: 在 `has_result` 之后加 `result.get("status") == "success"` 检查，否则进入 FAILED 分支。

### 1.3 `vision_obs.result` 结构规范化

**涉及文件**:
- `VISTA/vision_module/app/stages/grasp.py` (同上段)
- `orchestrator/orchestrator_service/ipc/protocol.py` — 消费侧解析

**改动**: VISTA 从 grasp server HTTP 响应的 `targets[0]` 提取首元素放入 `result.grasp`，而非透传完整 `{status, targets, grasp_count, ...}`。

### 1.4 Orchestrator 状态机新增 GRASP 状态

**涉及文件**: `orchestrator/orchestrator_service/runtime/state_machine.py`

**改动**: `FREEZE_BASE → GRASP → DONE`，GRASP 状态下发送 `make_grasp_req()`，等待 `vision_obs(status=RESULT_READY)`，提取 `result.grasp` 后发 UART 命令。

---

## 二、对当前消息结构的建议

### 2.1 `req_type` 与 `op` 语义重叠

**涉及文件**: `VISTA/vision_module/app/stage_controller.py`

**关键代码** (stage_controller.py:50-59):

```python
@staticmethod
def _request_type(req: VisionReq) -> str:
    payload = req.payload if isinstance(req.payload, dict) else {}
    req_type = str(getattr(req, "req_type", "") or payload.get("req_type") or "").strip().lower()
    if req_type in {"mode_request", "target_update", "keepalive"}:
        return req_type
    op = normalize_upper(req.op, "START")
    if req.is_stop() or op in {"START", "STOP"}:
        return "mode_request"
    return "target_update"  # ← RESPOND 也被归为此类，语义不准
```

**建议**: RESPOND 应独立为 `respond` 类型，或至少不归类为 `target_update`。

### 2.2 VistaApp 与 StageContext 状态双写

**涉及文件**:
- `VISTA/vision_module/app/app.py` — `self.current_stage/mode/session_id/req_id/epoch`
- `VISTA/vision_module/app/stage_controller.py` — `self._ctx.current_stage/mode/session_id/...`

**关键代码** (app.py:637-646):

```python
def _sync_runtime_from_stage_context(self, reason=""):
    ctx = self.stage_controller.context()
    self.current_stage = self._safe_stage_text(ctx.current_stage)
    self.current_mode = self._safe_mode_text(ctx.current_mode)
    self.target_name = ctx.target_name
    self.current_session_id = ctx.session_id
    self.current_req_id = ctx.req_id
    self.current_epoch = int(ctx.epoch)
    self.active_interaction_id = ctx.interaction_id
```

**建议**: VistaApp 直接通过 `self.stage_controller.context()` 读取，去除自有副本和同步函数。

### 2.3 effects 机制正式化

**涉及文件**:
- `VISTA/vision_module/app/stages/grasp.py` — 当前唯一使用者
- `VISTA/vision_module/app/stages/base.py` — 应新增 `emit_command()` 方法
- `VISTA/vision_module/app/stage_controller.py` — `_publish_effects()`

**关键代码** (grasp.py:146-155):

```python
def _remote_effect(op: str, payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "type": "PUBLISH_EVENT",
        "route": "remote_cmd",
        "payload": {"op": normalize_upper(op, "UNKNOWN"), **dict(payload or {})},
    }
```

**建议**: 在 `BaseStagePlan` 中添加 `emit_event(route, payload)`，去掉 dict 手写。

---

## 三、消息 ID 字段说明

| ID | 生成方 | 生命周期 | 关键代码 |
|----|--------|----------|----------|
| `session_id` | mobile_gateway | 多请求会话 | `protocol.py:260` — `str(payload.get("session_id") or ...)` |
| `req_id` | Orchestrator | 单次 vision_req | `protocol.py:644` — `req_id or _new_id("req")` |
| `epoch` | mobile_gateway | 会话内递增 | `protocol.py:261` — `int(payload.get("epoch", 0) or 0)` |
| `interaction_id` | VISTA | 单次 MICRO_ADJUST 交互 | `base.py:71-72` — `f"ia_{int(time.time() * 1000)}"` |

**interaction_id 的防御用途** (grasp.py:232-245):

```python
if ctx.interaction_id and req.interaction_id \
   and str(req.interaction_id) != str(ctx.interaction_id):
    return StageOutput(
        vision_obs=self.build_obs(ctx, status="FAILED",
            result={"reason": "interaction_id_mismatch"}),
        signals={"response": "ERROR", "reason": "interaction_id_mismatch"},
    )
```

---

## 四、Mode 切换 → Scheduler 重置机制

### 4.1 _compile_plan() 生成 plan

**涉及文件**: `VISTA/vision_module/backend/mode_controller.py`

**关键代码** (mode_controller.py:100-128):

```python
def _compile_plan(self, profile: ModeProfile) -> Dict[str, Any]:
    return {
        "mode": str(profile.name or "IDLE").strip().upper(),
        "routes": {
            "camera_frames":   {"policy": "slot", "scope": "backend"},
            "frame_meta":      {"policy": "slot", "scope": "stage"},
            "local_perception": {"policy": "slot", "scope": "stage"},
            "table_edge_obs":  {"policy": "slot", "scope": "stage"},
            "remote_result":   {"policy": "slot", "scope": "stage"},
            "runtime_status":  {"policy": "slot", "scope": "backend"},
            "remote_cmd":      {"policy": "event", "scope": "backend"},
            "remote_ack":      {"policy": "event", "scope": "backend"},
        },
        "capabilities": { ... },
    }
```

`scope=stage` → `collect_tick_input()` 收集；`scope=backend` → manager 间读写，不进入 tick。

### 4.2 scheduler.configure() 清空旧数据

**涉及文件**: `VISTA/vision_module/backend/scheduler.py`

**关键代码** (scheduler.py:65-81):

```python
def configure(self, plan, generation):
    self.active_plan = plan
    self.active_generation = generation
    self.routes = plan.get("routes")       # ← 激活新路由
    self.result_slots.clear()               # ← 清空旧 mode 数据
    self.event_latches.clear()
```

### 4.3 RuntimeSupervisor.reconcile() 启停 manager workers

**涉及文件**: `VISTA/vision_module/backend/runtime_supervisor.py`

**关键代码** (runtime_supervisor.py:366-375):

```python
def _apply_plan(self, plan, generation):
    ok = True
    ok = self._configure_camera(...) and ok
    ok = self._configure_predictor(...) and ok
    ok = self._configure_remote(...) and ok
    ok = self._configure_table_edge(...) and ok
    ok = self._configure_preview(...) and ok
    return ok
```

每次 mode 切换，不相关的 manager 被 `stop_runtime()` 停掉（`_worker_stop.set()` + `join()`），新 mode 的 manager 被 `start_runtime()` 启动。**旧 mode 的 worker 线程已完全退出，不会再向 scheduler 写数据。**

---

## 五、请求处理即时输出 vs tick 输出

### 5.1 handle_request 即时输出：多数无 payload

**涉及文件**: `VISTA/vision_module/app/app.py`

**关键代码** (app.py:693-700):

```python
def _apply_stage_output(self, output, now, force_send=False):
    if output is None or output.vision_obs is None:
        return False       # ← on_enter/on_update 多数只返回 signals，无 vision_obs
    if not force_send and (now - self.last_send_ts) < self._send_interval_s():
        return False       # ← 限速
    queued = self._send_obs(output.vision_obs)
```

### 5.2 tick 输出：manager 数据就绪后才产出

**涉及文件**: `VISTA/vision_module/app/app.py` + `VISTA/vision_module/backend/scheduler.py`

**关键代码** (app.py:815-827):

```python
def _tick_stage(self, now):
    tick_input = self.runtime.collect_tick_input(ts=now)
    # ↑ scheduler 收集 scope=stage 的 result_slots
    # mode 刚切换后，slots 为空 → tick_input.results = {}
    stage_output = self.stage_controller.tick(tick_input)
    self._apply_stage_output(stage_output, now=now)
```

`scheduler.configure()` 在 mode 切换时清空 slots，因此新 mode 的第一个 tick 必然没有旧数据。Manager workers 也是新启动的，第一个结果需要等 worker loop 产出。

---

## 六、effects 通道（StagePlan → Manager）

**目的**: GraspStagePlan 需要主动触发 RemoteManager 的 INIT / PREDICT / RELEASE 操作。

**完整流转**:

```
GraspStagePlan.tick()
  → StageOutput(effects=[{"type":"PUBLISH_EVENT", "route":"remote_cmd", ...}])
    → StageController._finalize_output()
      → _publish_effects()                              # stage_controller.py:288-302
        → runtime_service.publish_event(route, payload) # vision_engine.py:200-201
          → scheduler.publish_event(route, payload)     # scheduler.py:113-137
            → event_latches["remote_cmd"].append(...)    # 入队
              → RemoteManager._worker_loop()             # worker 线程
                → consume_event("remote_cmd")            # 消费
                  → _handle_command(cmd)                 # 执行 INIT/PREDICT/RELEASE
```

**关键代码** — 生产侧 (stage_controller.py:288-302):

```python
def _publish_effects(self, effects):
    for effect in effects:
        if effect.get("type") == "PUBLISH_EVENT":
            self._runtime_service.publish_event(
                effect["route"], effect["payload"])
```

**关键代码** — 消费侧 (remote/manager.py:525-537):

```python
def _worker_loop(self):
    while self._runtime_running and not self._worker_stop.is_set():
        cmd = scheduler.consume_event("remote_cmd")
        if isinstance(cmd, dict):
            ack = self._handle_command(cmd)
            self._publish_event("remote_ack", ack)
        self._publish_result("remote_result", self.result_summary())
        self._worker_stop.wait(timeout=self._worker_interval_s)
```

---

## 七、入向消息完整路由

```
JsonlInboundServer (TCP :9003)
  → drain() → queue → main loop
    → _handle_request_payload(payload)
      → VisionReq.from_dict(payload)    # protocol.py:123-139
        → _canonical_stage()            # 兼容 home_tag_req → RETURN
        → _canonical_op()               # 兼容 mode=IDLE → STOP
      → StageController.handle_request(req)
        → _request_type(req)            # keepalive|mode_request|target_update
        → _sync_request_context(req)    # req → StageContext
        ┌─ keepalive → 仅返回 signals
        ├─ idempotent → on_update()
        ├─ STOP → on_stop() → transition_to(IDLE)
        ├─ RESPOND → on_respond()
        ├─ stage变化 → transition_to() → on_enter()
        ├─ START同stage → transition_to() → on_enter() (restart)
        └─ UPDATE → on_update()
      → _apply_context_mode(reason)     # 可能触发 mode 切换
        → ModeController.switch_mode()
          → _compile_plan(profile) → plan
          → VisionEngine.apply_mode_plan(plan, generation)
            → scheduler.configure()     # 更新 routes + 清空 slots/events
            → RuntimeSupervisor.reconcile()  # 启停 manager workers
```

---

## 八、出向消息生产过程

```
主循环 run() 每帧:
  ┌─ req_server.drain() → 处理所有排队请求
  │    └─ _handle_request_payload() → 即时发送 (如果有 vision_obs)
  │
  └─ _tick_stage(now)
       ├─ scheduler.collect_tick_input(ts=now)
       │    └─ 聚合 scope=stage 的 slots → StageTickInput.results
       ├─ StageController.tick(tick_input)
       │    └─ plan.tick() → StageOutput
       └─ _apply_stage_output(stage_output)
            ├─ 检查 vision_obs is None → return
            ├─ 检查 send_hz 限速 → return
            └─ _send_obs(vision_obs)
                 └─ JsonlClientSender.send()
                      └─ queue → worker线程 → TCP JSONL → :9002
```

---

## 九、Orchestrator 底盘控制全链路 + GRASP UART 集成规划

分析日期：2026-05-04
来源：Orch-Code-Inspect（代码探索 Agent）

### 9.1 最小阅读范围

理解底盘控制链路需要阅读的文件（按顺序）：

| # | 文件 | 关键行 | 内容 |
|---|------|--------|------|
| 1 | `orchestrator/orchestrator_service/runtime/service.py` | 788-806, 981-1117, 1476 | 主循环 tick、vision_obs 流入、motion 流出 |
| 2 | `orchestrator/orchestrator_service/runtime/state_machine.py` | 200-226, 881-930, 1555, 1843-1864 | 状态分发、底盘状态 tick、观测新鲜度过滤 |
| 3 | `orchestrator/orchestrator_service/control/docking_controller.py` | 50-79, 107-163 | PID 参数、per-mode 控制逻辑 |
| 4 | `orchestrator/orchestrator_service/control/pid.py` | 全文(短) | 离散 PID（死区、积分限幅、输出限幅） |
| 5 | `orchestrator/orchestrator_service/runtime/controller.py` | 127-147, 238, 254, 307-355 | MotionController: 各状态命令、fallback 转向 |
| 6 | `orchestrator/orchestrator_service/bridge/simple_car_protocol.py` | 67-99, 103 | CmdVel → UART 行映射 |
| 7 | `orchestrator/orchestrator_service/bridge/uart_bridge.py` | 95, 140-154, 175, 218 | UART 收发线程、latest-command-override |
| 8 | `orchestrator/orchestrator_service/ipc/protocol.py` | 275-288, 541-607, 622-649 | TaskAck, CarState, CmdVel, make_vision_req |
| 9 | `orchestrator/orchestrator_service/config/schema.py` | 30-34, 43-68, 166-200 | SerialConfig, RuntimeConfig, ControlThresholds, CarMotionConfig |

### 9.2 底盘控制全链路（简要）

```
TCP :9002 → vision_obs
  |
  +- _drain_vision_msgs()           [service.py:981]
  |   解析 VisionObsEnvelope，提取 table_edge_obs/target_obs/home_tag_obs
  |   每种取最新一条 → RuntimeContext
  |
  +- _drain_uart_feedback()         [service.py:866]
  |   读 STM32: STATE / ESTOP 行 → CarState
  |
  +- _drain_task_cmds()             [service.py:898]
  |   读 mobile_gateway: FIND / RETURN / STOP → TaskCmd
  |
  +- core.tick() @ 10Hz             [state_machine.py:200]
  |   +- _check_safety_interlock()  → 障碍物检测
  |   +- dispatch 到当前 state._tick_*()
  |   |   +- _fresh_table_obs()     → 过滤超过1s的过期观测
  |   |   +- 计数器迟滞               → N帧连续满足才切换
  |   |   +- 超时检测
  |   |   +- MotionController       → PID / fallback 比例转向
  |   |   +- 返回 MotionDecision(CmdVel)
  |   +- 状态转移判断
  |
  +- _flush_pending_msgs()          [service.py:1160]
  |   发送 vision_req 给 VISTA
  |
  +- _emit_motion(decision)         [service.py:1476]
      mapper.from_cmd_vel()          [simple_car_protocol.py:67]
        clamp + 3位小数格式化 → "MODE xxx\nVEL vx vy wz hold_ms\n"
        → uart.send_car_command()    [uart_bridge.py:95]
          → latest-command-override
            → ser.write(bytes)       [/dev/ttyHS1 @ 115200]
```

### 9.3 底盘控制除 UART 发送外的其他工作

| 工作 | 位置 | GRASP 是否需要 |
|------|------|---------------|
| 观测解析+类型路由+去重 | `service.py:_drain_vision_msgs` | **需要** — 解析 `vision_obs.result.grasp` |
| 观测新鲜度过滤(>1s丢弃) | `state_machine.py:_fresh_table_obs` | **需要** — grasp 结果也有时效 |
| 计数器迟滞(N帧连续满足) | `state_machine.py` 各 `_tick_*` | **不需要** — grasp 是单次结果，非连续帧 |
| 3轴 PID + 死区 + 积分限幅 + 速率限制 | `docking_controller.py` | **不需要** — grasp 是离散位姿，非连续速度 |
| fallback 比例转向 | `controller.py:_fallback_table_cmd` | **不需要** — 无 fallback 场景 |
| 障碍物安全联锁 | `state_machine.py:_check_safety_interlock` | 可复用 |
| 超时管理 | 各 tick 超时检查 | **需要** — grasp 整体超时 |
| vision_req 生命周期管理 | `_flush_pending_msgs` | **需要** — GRASP START + RESPOND |
| 重试计数 | COARSE_ALIGN → DOCK_RETRY 模式 | **需要** — reposition 重试 |
| 日志记录(timeline/ipc/cmd_vel) | state_machine.py | **需要** |
| task_ack 发送 | service.py | **需要** — 通知 mobile_gateway |

### 9.4 STM32 机械臂协议

用户提供的 STM32 机械臂指令：

```
HELP                               → 打印帮助
RESET                              → 机械臂复位，返回 "OK RESET"
POSE x y z pitch roll claw time    → 移动机械臂，返回 "OK POSE..." 或 "ERR IK..."
```

**参数说明**：

| 参数 | 单位 | 类型 | 来源 |
|------|------|------|------|
| `x` | cm | int | grasp result `x_cm` 四舍五入 |
| `y` | cm | int | grasp result `y_cm` 四舍五入 |
| `z` | cm | int | grasp result `z_cm` 四舍五入 |
| `pitch` | 度 | int | grasp result `pitch_deg` 四舍五入 |
| `roll` | 度 | int | grasp result `roll_deg` 归一化到 [-90, +90]（二指夹爪 +-180 等价） |
| `claw` | 角度(deg) | int | grasp result `gripper_width_cm` 通过查找表转换为张合角度 |
| `time` | ms | int | 机械臂运行时间，暂缺，后续写死或从上层传入 |

**roll 归一化**: `roll = round(roll_deg) % 180`; 若 `> 90` 则 `roll = roll - 180`

**claw 转换**: `gripper_width_cm → 夹爪张合角度`，查找表由 STM32 侧提供。当前先搭建转换框架（预留 `width_to_angle()` 函数，内部用临时映射表占位）。

**POSE 命令示例**:
```
POSE 15 0 12 0 0 40 1000    ← x=15cm y=0cm z=12cm pitch=0 roll=0 claw=40 time=1000ms
```

### 9.5 GRASP UART 命令集成计划

`SimpleCarMapper` 当前只处理底盘命令（MODE/VEL/STOP/BRAKE）。机械臂命令语义不同（位姿而非速度），**新增独立的 mapper/encoder**，在 `_emit_motion()` 中根据命令类型分发。

#### 新增文件

| 文件 | 职责 |
|------|------|
| `bridge/arm_protocol.py` | 机械臂协议编码：POSE 行格式化、RESET、HELP |
| `utils/target_utils.py` | 已完成：`target_to_class_id()` |
| `utils/grasp_utils.py` (待定) | grasp result → POSE 参数转换（roll 归一化、claw 查表占位） |

#### 修改文件

| 文件 | 改动 |
|------|------|
| `ipc/protocol.py` | 新增 `make_grasp_req(target, class_id, ...)` |
| `runtime/state_machine.py` | 新增 `GRASP` 状态 + `_tick_grasp()` |
| `runtime/service.py` | `_emit_motion()` 支持 arm 命令分发；`_drain_vision_msgs()` 提取 `result.grasp` |

#### GRASP 状态 tick 流程（草案）

```
_tick_grasp():
  |
  +- 初次进入 → make_grasp_req(target, class_id)
  |             → vision_req(stage=GRASP, op=START)
  |             → substate: AWAITING_RESPOND
  |
  +- AWAITING_RESPOND:
  |   收到 WAITING_RESPONSE → RESPOND(decision=ACCEPT)
  |   substate: AWAITING_RESULT
  |
  +- AWAITING_RESULT:
  |   超时 → ERROR_RECOVERY
  |   |
  |   +- status=RESULT_READY → result.grasp
  |   |     +- grasp_utils: cm→int, roll归一化, claw查表
  |   |        → arm_protocol.encode_pose(x,y,z,pitch,roll,claw,time)
  |   |        → uart.send_arm_command("POSE ...")
  |   |        → substate: AWAITING_ARM
  |   |
  |   +- status=RUNNING + reposition_hint
  |   |     +- 微调底盘 → 重试计数++
  |   |        → if >3: ERROR_RECOVERY
  |   |        → substate: AWAITING_RESPOND (重新发 START)
  |   |
  |   +- status=FAILED → 按 reason:
  |         no_detection → SEARCH_TARGET_INIT
  |         no_grasp_detected → ERROR_RECOVERY
  |
  +- AWAITING_ARM:
      收到 STM32 "OK POSE" → DONE → task_ack(DONE)
      收到 STM32 "ERR IK"  → reposition 重试(如果还有余额)
      超时 → ERROR_RECOVERY
```

#### GRASP 与底盘状态的关键差异

| 维度 | 底盘状态 | GRASP 状态 |
|------|---------|-----------|
| 控制模式 | 连续闭环(PID 每 tick) | 离散开环(POSE 单发等回复) |
| 观测来源 | VISTA 连续帧 | grasp server HTTP 单次结果 |
| 迟滞机制 | N帧计数器 | 不需要 |
| UART 命令 | MODE/VEL (latest-override 可覆盖) | POSE (单次发送, 不可覆盖) |
| UART 回复 | STATE (每帧) | OK POSE / ERR IK (单次) |
| 重试方式 | 底盘后退 → DOCK_RETRY | 微调 → reposition → 重发 START |
| 所需控制器 | PID (3轴) + slew-rate + fallback | 仅 grasp_utils 转换函数 |
