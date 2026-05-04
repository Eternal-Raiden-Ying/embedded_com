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
