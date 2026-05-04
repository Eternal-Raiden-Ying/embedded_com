# TODO.md — 项目总后续工作跟踪

更新时间：2026-05-02

状态约定：
- `todo`：未开始
- `doing`：正在推进
- `done`：已完成
- `blocked`：存在外部依赖或前置条件未满足

---

## 项目当前状态概览

| 模块 | 状态 | 关键未完成项 |
|------|------|-------------|
| mobile_gateway | MVP 可用，小程序已稳定 | MQTT broker 生产化、ASR 管线（优先级下调至 P1） |
| Orchestrator | 状态机升级完成，桌边-沿边搜索-目标锁定主链稳定 | **GRASP 阶段未接入状态机闭环** |
| VISTA | Detect/Remote/Runtime 线全部 DONE | GraspStagePlan status 检查修复、端到端联调 |
| grasp server | 推理管线完成，输出字段对齐确认 | 输出协议冻结（加 operating_time） |
| Voice/ASR | 已从板端归档 | 手机端/云端重建 ASR→NLU→mobile_cmd 管线（P1） |

---

## P0 — Grasp 串联进入主链路

对应文档：[docs/grasp_protocol_analysis.md](docs/grasp_protocol_analysis.md)、[SYSTEM_UPGRADE_DELIVERY.md](SYSTEM_UPGRADE_DELIVERY.md)

### 1. Grasp Server 输出协议冻结
- 状态：`doing`
- 优先级：`P0`
- 已确认：
  - 输出字段（11 个）足够驱动机械臂
  - 坐标系转换在 server 侧完成（手眼标定），输出为 robot 系
  - 板端请求已对齐 class_id-only 模式（已确认，无需修改）
- 待完成：
  - [ ] 新增 `operating_time` 字段（预留，可 null）
  - [ ] 确认 `reason` 枚举不变
  - [ ] 通知板端侧格式已冻结

### 2. target → class_id 映射函数
- 状态：`todo`
- 优先级：`P0`
- 内容：
  - 新建 `orchestrator/orchestrator_service/utils/target_utils.py`
  - 实现 `target_to_class_id(target: str) -> int`，基于 COCO80 类名查表
  - 独立函数包装，后续换 YOLO 类别集只需改此函数
- 阻塞项：无

### 3. VISTA 侧 GraspStagePlan 修复
- 状态：`todo`
- 优先级：`P0`
- 内容：
  - `GraspStagePlan.tick()` 增加 `status=="success"` 判断
  - 当前仅检查 `has_result`，会误将 `reposition_required` 当作 `RESULT_READY`
- 参考：`VISTA/vision_module/app/stages/grasp.py:425-441`

### 4. Orchestrator 侧 GRASP 状态实现
- 状态：`blocked`（依赖 Item 2 + Item 1）
- 优先级：`P0`
- 内容：
  - 新建 `make_grasp_req()` 辅助函数
  - 状态机新增 `GRASP` 状态：`FREEZE_BASE → GRASP → DONE`
  - 消费 `vision_obs.result.targets[0]` 获取最优抓取位姿
  - 通过 UART 向 STM32 发送机械臂控制命令（协议待定）
- 参考：`orchestrator/orchestrator_service/runtime/state_machine.py`

### 5. 端到端集成测试
- 状态：`blocked`（依赖 Item 4）
- 优先级：`P0`
- 内容：全链路 `mobile_cmd → search → lock → grasp → DONE`

---

## P1 — Mobile Gateway 生产化（优先级下调）

小程序框架已稳定可用，不再做架构调整。

### 6. MQTT Broker 选型与部署
- 状态：`todo` | 优先级：`P1`
- 参考：`docs/mobile_control_next_steps.md` Section 1

### 7. 小程序端 ASR → 结构化命令管线
- 状态：`todo` | 优先级：`P1`
- 参考：`docs/mobile_control_next_steps.md` Section 3

### 8. 视障用户音频反馈
- 状态：`todo` | 优先级：`P1`
- 参考：`docs/mobile_control_next_steps.md` Section 4

### 9. 生产加固
- 状态：`todo` | 优先级：`P1`
- 参考：`docs/mobile_control_next_steps.md` Section 5

---

## P1 — 代码结构优化 + 文档同步

### 10. vision_develop TRACK_TABLE 合入评估
- 状态：`todo` | 优先级：`P1`
- 发现：vision_develop 的 VISTA 在 TRACK_TABLE 上领先 up2date 一个工作周期
- 待评估：是否合入？还是 up2date 已有不同实现覆盖？

### 11. VISTA ReadMe.md 文档同步
- 状态：`todo` | 优先级：`P1`
- 发现：ReadMe.md 将 DEPTH_PERCEPTION/TABLE_EDGE_PERCEPTION 列为当前 mode（实际是扩展方向），日志文件名引用已废弃

### 12. `release_cooldown_s` 决策
- 状态：`todo` | 优先级：`P1`
- 发现：mode profile 中声明但 runtime 不生效，需实现或移除

### 13. VISTA preview/table_edge 设计一致性审查
- 状态：`todo` | 优先级：`P1`
- 内容：审查 `preview/`、`table_edge_*` 与 `vision_engine.py`/`mode_controller.py` 的设计一致性
- 注意：开发新代码前需讨论框架设计

### 14. vision_module vs vision_module_v2 关系清理
- 状态：`todo` | 优先级：`P1`

### 15. 日志统一
- 状态：`todo` | 优先级：`P1`
- 按 `LOG_STANDARD.md` 对齐各模块

---

## P2 — 远期 / 可选

### 16. Vision Module v2 深度桌边提取
- 状态：`blocked`（等待 Offline_Edge_Test 验证结果）
- 优先级：`P2`

### 17. 多入口 Control Plane（远期架构）
- 状态：`todo` | 优先级：`P2`
- 参考：`docs/mobile_control_next_steps.md` Section 6

---

## 状态变更规则

1. 每次推进进度时更新对应条目
2. 状态变更时同步更新 `next_todo.md`
3. `blocked` 条目需标注阻塞原因和解除条件
4. 完成条目不删除，标记为 `done` 并保留最后更新时间
