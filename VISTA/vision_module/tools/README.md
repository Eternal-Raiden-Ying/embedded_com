# vision_module 工具目录 (vision_module tools)

此目录预留用于手动调试脚本和操作员运行的辅助工具。

当前状态：

- 历史脚本（如 `debug_send_req.py`、`debug_recv_obj.py`、`debug_protocol_tools.py` 和 `demo_camera.py`）仍保留在 `test/` 中。
- 在本次清理过程中特意没有移动它们，以便板端的现有使用习惯和 Shell 命令能够继续正常工作。

推荐的下一步工作：

- 每次将一个脚本逐步从 `test/` 移入此目录。
- 移动脚本时，在原路径保留一个小的兼容性垫片（compatibility shim）。
- 自动化单元测试和回归测试请保留在 `test/` 中。
