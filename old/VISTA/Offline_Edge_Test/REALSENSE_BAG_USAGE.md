# RealSense Bag Usage

## 当前结论

已验证当前工作区中的 [desk_scene.bag](d:/55495/workspace/embedded_project/embedded_com/desk_scene.bag) 可以正常被 `pyrealsense2` 打开并回放，但这份 bag 当前只包含 `color` 流，不包含真正的 `depth` 图像流。

本地诊断结果：

- 设备来源：`D435`
- 设备传感器：`Stereo Module`、`RGB Camera`
- 实际录入 bag 的流：只有 `stream.color / rgb8 / 1280x720 / 15fps`

这说明：

- 录制设备本身是有深度模组的
- 但录制时没有把 depth 流真正写进 bag
- 当前更像是 “录制时只开了 RGB” 而不是 “读取脚本漏读了 depth”

## 读取脚本

脚本路径：

- [read_realsense_bag.py](d:/55495/workspace/embedded_project/embedded_com/VISTA/Offline_Edge_Test/read_realsense_bag.py)
- [record_realsense_bag.py](d:/55495/workspace/embedded_project/embedded_com/VISTA/Offline_Edge_Test/record_realsense_bag.py)

### 基础读取

```powershell
conda run -n realsense_py310 python VISTA\Offline_Edge_Test\read_realsense_bag.py --bag desk_scene.bag --max-frames 5 --save-index 1
```

### 带预览窗口

```powershell
conda run -n realsense_py310 python VISTA\Offline_Edge_Test\read_realsense_bag.py --bag desk_scene.bag --preview
```

### 如果 bag 中有 depth，同时直接跑 TableEdgeDetector

```powershell
conda run -n realsense_py310 python VISTA\Offline_Edge_Test\read_realsense_bag.py --bag desk_scene.bag --run-detector --calib-json VISTA\Offline_Edge_Test\calib.json
```

## 这次测试结果为什么不能直接跑 TableEdgeDetector

因为 `TableEdgeDetector` 需要 `16-bit depth png`，而当前 `desk_scene.bag` 没有录下 depth 帧。

所以现在的现象是：

- 可以导出彩色图
- 无法导出深度图
- `--run-detector` 会自动提示跳过原因，而不是报错崩掉

## 在 RealSense Viewer 里如何重新录制 depth + color

建议按下面顺序录制：

1. 打开 `RealSense Viewer`
2. 连接 D435，确认设备已经被识别
3. 在左侧设备面板里，先展开并开启：
   - `Stereo Module / Depth`
   - `RGB Camera / Color`
4. 先不要急着录制，先看实时预览里是否真的已经同时出现：
   - depth 画面
   - color 画面
5. 确认 depth 正在实时刷新后，再点击 `Record to File`
6. 录制结束后停止保存，生成新的 `.bag`
7. 用本脚本先检查新 bag：

```powershell
conda run -n realsense_py310 python VISTA\Offline_Edge_Test\read_realsense_bag.py --bag your_new_file.bag --max-frames 3 --run-detector
```

## 重新录制时的注意事项

- 录制前必须先确认 depth 流已经真的在 Viewer 中启动，而不是只有设备连接成功
- 如果只打开了 RGB 预览再去录制，bag 很可能就只保存 color
- 重新录制后第一件事不要直接跑算法，先用本脚本检查流列表里是否出现：
  - `stream.depth`
  - `stream.color`
- 如果新 bag 仍然只有 color，优先排查：
  - depth 开关是否在录制前已开启
  - depth 是否有实时画面
  - 录制时是否切换过 profile
  - 是否误用了只录单流的导出路径

## 在插着 D435 的设备上直接录制的具体做法

### 方法 1：用 RealSense Viewer 录制

如果设备上装了 `realsense-viewer`，推荐优先用这个方法。

#### Linux 设备

```bash
realsense-viewer
```

#### Windows 设备

直接打开 `RealSense Viewer`

#### 录制步骤

1. 插上 D435，确认系统识别成功
2. 打开 `RealSense Viewer`
3. 在左侧设备面板展开：
   - `Stereo Module`
   - `RGB Camera`
4. 在 `Stereo Module` 里先启用 `Depth`
   推荐先用：
   - `640x480`
   - `30 fps`
5. 在 `RGB Camera` 里启用 `Color`
   推荐先用：
   - `1280x720`
   - `30 fps`
6. 先确认界面里同时出现：
   - 深度预览
   - 彩色预览
7. 这一步很关键：
   必须是在两个流都已经开始实时刷新之后，再点击 `Record to File`
8. 选择保存路径，开始录制
9. 录完后停止
10. 立刻用下面这条命令检查录出来的 bag 是否真的包含 depth：

```powershell
conda run -n realsense_py310 python VISTA\Offline_Edge_Test\read_realsense_bag.py --bag 你的新文件.bag --max-frames 3
```

你应该在输出里看到：

- `stream.depth`
- `stream.color`

如果只看到 `stream.color`，说明录制时 depth 没真正录进去。

### 方法 2：用 Python 脚本直接录制

这个方法适合：

- 设备上没有方便的图形界面
- 你想固定参数直接录
- 你想把录制流程脚本化

命令示例：

```powershell
conda run -n realsense_py310 python VISTA\Offline_Edge_Test\record_realsense_bag.py --output new_rgb_depth.bag --duration 10 --preview
```

或 Linux 上：

```bash
conda run -n realsense_py310 python VISTA/Offline_Edge_Test/record_realsense_bag.py --output new_rgb_depth.bag --duration 10 --preview
```

这条脚本会同时开启：

- `Depth: 640x480@30`
- `Color: 1280x720@30`

并直接录成一个 `.bag`。

录完之后，马上检查：

```powershell
conda run -n realsense_py310 python VISTA\Offline_Edge_Test\read_realsense_bag.py --bag new_rgb_depth.bag --run-detector
```

## 推荐的下一步

拿一份同时包含 `depth + color` 的 bag 重新跑：

```powershell
conda run -n realsense_py310 python VISTA\Offline_Edge_Test\read_realsense_bag.py --bag new_depth_color.bag --run-detector --preview
```

这样你就能一次性验证三件事：

- bag 是否能正常打开
- depth png 是否能正常导出
- `TableEdgeDetector` 对这份深度数据是否能跑通

## 当前环境补充说明

当前 `realsense_py310` 环境里已经可以正常 `import pyrealsense2`，但还没有安装 `scikit-learn`。

而 [TableEdgeDetector.py](d:/55495/workspace/embedded_project/embedded_com/VISTA/Offline_Edge_Test/TableEdgeDetector.py) 依赖：

- `sklearn.linear_model.RANSACRegressor`

所以当你后续拿到真正包含 depth 的 bag，并且想直接在 `realsense_py310` 里跑 `--run-detector` 时，还需要先补这个依赖，例如：

```powershell
conda activate realsense_py310
pip install scikit-learn
```
