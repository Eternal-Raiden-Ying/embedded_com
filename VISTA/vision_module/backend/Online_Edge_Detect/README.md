# Online_Edge_Detect

在线桌边检测版本，保留原有 `Offline_Edge_Test` 不动。

这版目标：

- 直接在板子上连接 RealSense 相机进行实时桌边检测
- 不依赖 `sklearn`
- 结构和运行方式参考 `vision_module`
- 可选把检测结果按 `table_edge_obs` JSON 输出到 TCP/UDS

## 目录说明

- `board_config.py`
  运行参数和环境变量入口
- `detector.py`
  轻量版在线桌边检测器
- `stream_source.py`
  RealSense 在线采集源，支持 live 和 bag
- `app.py`
  在线主程序
- `synthetic_smoke_test.py`
  不接相机时的算法快速测试

## 直接在板子上运行

```bash
python3 VISTA/Online_Edge_Detect/app.py
```

如果板子没有图形桌面，建议先关掉预览：

```bash
export EDGE_PREVIEW=0
python3 VISTA/Online_Edge_Detect/app.py
```

## 推荐环境变量

```bash
export EDGE_CALIB_JSON=/home/aidlux/2026/VISTA/Offline_Edge_Test/calib.json
export EDGE_DEPTH_WIDTH=424
export EDGE_DEPTH_HEIGHT=240
export EDGE_DEPTH_FPS=15
export EDGE_COLOR_ENABLED=1
export EDGE_COLOR_WIDTH=1280
export EDGE_COLOR_HEIGHT=720
export EDGE_COLOR_FPS=15
export EDGE_PREVIEW=1
```

## 可选输出到 orchestrator

```bash
export EDGE_OUT_TRANSPORT=tcp
export EDGE_OUT_HOST=127.0.0.1
export EDGE_OUT_PORT=9002
```

这样在线检测结果会按 `table_edge_obs` 格式发出去。

## 本地不接相机时的快速检查

```bash
python3 -m VISTA.Online_Edge_Detect.synthetic_smoke_test
```

## 板端先做离线验证的推荐流程

先不要上来就录 bag。更稳的顺序是：

1. 用实时相机抓一张深度图

```bash
python3 VISTA/Offline_Edge_Test/capture_live_depth_snapshot.py --preview
```

2. 用在线版轻量检测器离线跑这张深度图

```bash
python3 VISTA/Online_Edge_Detect/offline_depth_png_test.py --depth-png 你的depth.png
```

这样你会得到：

- 一份检测结果 JSON
- 一张带 ROI 和结果文字的 preview PNG

先确认离线结果合理，再去跑在线流。
