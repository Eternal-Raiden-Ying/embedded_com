# next_todo.md — 待完成事项

更新时间：2026-05-05

已完成工作记录已移交至 [docs/handover.md](docs/handover.md)。

---

## P0 阻塞项

- **端到端测试**: 依赖板端测试 skill 就位（真车联调：mobile_cmd → search → lock → grasp → DONE）
- **VISTA 端到端验证**: grasp.py 修改已被 Orchestrator 消费的验证（需 VISTA 运行环境）

## P1

- effects 正式化、req_type 语义清理、状态双写消除、TRACK_TABLE 合入评估、文档同步、release_cooldown_s 决策
- Mobile Gateway 生产化（MQTT broker、ASR 管线、盲人反馈、生产加固）

详见 [TODO.md](TODO.md) items 7-17。
