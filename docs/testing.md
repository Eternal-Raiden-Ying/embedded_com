# 测试体系 (Testing)

默认的测试策略是**“仅限必要测试”**。避免在 Windows 主机开发期间运行广泛、缓慢且依赖真实硬件的集成测试。

---

## 1. Windows 主机测试环境 (Windows Host Environment)

只允许使用以下环境执行测试：
```powershell
D:\anaconda\Anaconda\envs\embed_sc171\python.exe
```

在运行前，必须配置环境变量 `PYTHONPATH`：
```powershell
$env:PYTHONPATH="D:\55495\workspace\embedded_com;D:\55495\workspace\embedded_com\orchestrator;D:\55495\workspace\embedded_com\VISTA"
```

严禁直接使用 Base Anaconda 环境、MSYS2 的 Python 或其他未声明的默认 `python` 命令。

---

## 2. 必要测试项 (Necessary Tests)

必要测试是为了保障系统核心契约不受破坏：
*   **安全门锁门控 (Safety Gate)**：当车辆移动路径被阻挡或丢帧超过阈值时，底盘必须立刻停止。
*   **紧急刹车机制 (Emergency STOP)**：急停指令 `STOP` 拥有最高优先级，绝不能被后续排队的速度指令覆盖。
*   **视觉状态同步 (Vision State Sync)**：保证 Orchestrator 状态机与 VISTA Perceptual Stage 之间的模式同步不会产生时序滑移或失步。
*   **配置危险校验 (Config Override)**：保证危险的默认调参值不会静默应用生效。
*   **抓取微调判定 (Grasp Reposition)**：保证机械臂抓取前的车辆微调不会产生超调。
*   **观测路由隔离 (VISTA Observation Router)**：确保控制级观测数据具有独立通道，不会被大体积诊断日志堵塞丢帧。
*   **底层协议契约 (Protocol Contract)**：保持 `MODE`、`V`、`STOP`、`SSTOP` 串口协议以及视觉消息封包结构稳定。

具体的测试过滤和收集策略配置在 `pytest.ini` 中。

---

## 3. 主机端最小校验指令 (Minimal Host Commands)

```powershell
# 1. 静态语法检查与编译检查
D:\anaconda\Anaconda\envs\embed_sc171\python.exe -m compileall -q common orchestrator\orchestrator_service VISTA\vision_module

# 2. 运行核心单元测试
D:\anaconda\Anaconda\envs\embed_sc171\python.exe -m pytest tests\orchestrator tests\common VISTA\vision_module\test\test_observation_router.py VISTA\vision_module\test\test_stage_contract.py -q --tb=short --disable-warnings
```
**注意**：不要在后台随意执行未过滤的全局 `pytest tests`。

---

## 4. 手动与历史测试脚本 (Manual Tests)

手动、历史调试或交互式验证脚本统一存放在以下目录：
```text
scripts/manual/
```
例如：
*   `scripts/manual/manual_grasp.py`
*   `scripts/manual/legacy_tests/`

这些脚本在执行时必须排除在 pytest 自动收集范围之外。

---

## 5. 硬件依赖测试项 (Hardware Tests)

凡是需要真实相机硬件、RealSense SDK、开发板 QNN 二进制库或 aarch64 原生编译模块（如 `fast_cam`）的测试项，在 Windows 主机端均应被**跳过（Skip）**。

处理规范：
*   脚本文件以 `manual_` 命名开头。
*   使用 `@pytest.mark.hardware` 装饰器进行标记。
*   在测试头部使用平台断言进行动态跳过：
    ```python
    import platform
    import pytest
    if platform.system() == "Windows" or platform.machine() != "aarch64":
        pytest.skip("Hardware test skipped on current host platform", allow_module_level=True)
    ```

aarch64 原生动态库放置于：
```text
VISTA/vision_module/libs/aarch64/
```
Windows 主机端测试切勿直接对其进行 `import`。

---

## 6. 开发板物理测试规范 (Board Tests)

在主机通过自动化契约测试后，可以布设到 SC171 真实开发板上运行硬件调试。板端测试可以使用板载 Python、相机接口、物理串口以及特有的 QNN 硬件加速库。

板端实机验证标准顺序：
1.  以指定的运行 Profile 启动完整机器人软件栈。
2.  检查本地 `/tmp/robot_stack/` 各 UDS 接口文件是否正常绑定并就绪。
3.  通过网关调试指令核对 `vision_req` 与 `vision_obs` 的通信周期。
4.  在安全测试架上举升小车，验证下发的 `MODE`、`VEL`、`STOP` 协议是否完全符合物理轮向预期。
5.  测试紧急断电与急停打断功能，保证底盘控制回路的安全性。

请勿将板端调试过程产生的特定环境配置文件混入主机的清理提交中。
