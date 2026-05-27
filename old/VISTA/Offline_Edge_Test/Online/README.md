# Online

这是 `Offline_Edge_Test` 目录下的在线桌边检测版本。

这样目录职责就是：

- `Offline_Edge_Test/`
  - 抓取实时单帧
  - 离线验证 depth png
  - 原始离线检测器
- `Offline_Edge_Test/Online/`
  - 持续读取在线深度流
  - 实时跑桌边检测
  - 可选发出 `table_edge_obs`

## 板端直接运行

```bash
cd ~/2026/test/Offline_Edge_Test/Online
python3 app.py
```

如果板子没有图形环境：

```bash
export EDGE_PREVIEW=0
python3 app.py
```
