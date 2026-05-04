# next_todo.md — 下一步工作

更新时间：2026-05-02

---

## 已完成（本会话）

- [x] 审计全部 .md 文档 → `TODO.md`
- [x] 对比 simulate_client v1/v2 → 以 v2 为 baseline
- [x] VISTA 全部 DONE 确认；grasp server 推理管线确认
- [x] 板端代码与 grasp server class_id 对齐确认（一致，无需改）
- [x] 全链路协议分析 → `docs/grasp_protocol_analysis.md`
- [x] grasp server 输出字段确认（11 字段 + 新增 operating_time 预留）

## 立即进行

### 1. class_id 映射函数
- 文件：`orchestrator/orchestrator_service/utils/target_utils.py`（新建）
- 内容：`target_to_class_id(target: str) -> int`，COCO80 查表
- 理由：`task_cmd.target="apple"` → `vision_req.payload.class_id=47`
- 暂用 `target_aliases.json` 存放映射或直接硬编码 COCO80 常量

### 2. Grasp server 侧
- [ ] 确认 `_build_protocol_target()` 输出的 11 字段不变
- [ ] 新增 `operating_time` 字段（预留，可 null，后续写死或另处理）
- [ ] 输出格式冻结后通知板端侧

### 3. VISTA 侧小修
- [ ] `GraspStagePlan.tick()` 加 `status=="success"` 判断（当前仅检查 has_result，会误将 reposition_required 当成功）
- [ ] `docs/grasp_protocol_analysis.md` → 精简为 `docs/grasp_protocol.md`（只保留最终结论）

### 4. Orchestrator 侧
- [ ] 新建 `make_grasp_req()` 辅助函数
- [ ] 状态机新增 `GRASP` 状态（`FREEZE_BASE → GRASP → DONE`）
- [ ] 消费 `vision_obs.result.targets[0]` 获取抓取位姿

## 设计决策（已确认）

| 问题 | 结论 |
|------|------|
| grasp 输出字段 | 11 字段足够 + operating_time 预留 |
| position/angle frame 标注 | 保留或删除均可 |
| class_id 来源 | target→class_id 查表（COCO80），独立函数 |
| 板端请求是否 class_id | 已是，对齐 |
| 小程序优先级 | 下调，框架稳定不再改动 |
| grasp server 输出格式 | 当前格式即可，VISTA 透传，Orchestrator 解析 |

## 阻塞项

| 阻塞 | 原因 | 解锁条件 |
|------|------|---------|
| Orchestrator GRASP 实现 | 等 class_id 映射就位 | Item 1 完成 |
| 端到端测试 | 等 GRASP 状态 + 机械臂协议 | Item 4 + STM32 协议 |

## 下一步分工

| 任务 | 角色 |
|------|------|
| target→class_id 映射函数 | 功能开发 |
| grasp server operating_time + 冻结 | 用户 |
| VISTA status 检查修复 | 功能开发 |
| Orchestrator GRASP 状态 | 功能开发（等 class_id 映射就位） |
