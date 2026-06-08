# ROI simplified mapping patch

本补丁只处理视觉 ROI 逻辑，不修改 YOLO 模型/postprocess，不修改控制状态机。

## 1. 简化后的参数

保留的核心参数：

```yaml
yolo_table_roi_scale_x: 0.50
yolo_table_roi_scale_y: 0.50
rgb_depth_mapping_mode: centered_scale
rgb_fov_in_depth_scale_x: 0.75
rgb_fov_in_depth_scale_y: 0.75
rgb_depth_center_offset_x: 0.0
rgb_depth_center_offset_y: 0.0
yolo_table_bbox_hold_frames: 8
```

删除/不再使用的复杂参数：

```yaml
yolo_table_roi_expand_x_ratio
yolo_table_roi_expand_y_ratio
yolo_table_roi_min_w
yolo_table_roi_min_h
yolo_table_roi_max_w_ratio
yolo_table_roi_max_h_ratio
rgb_to_depth_view_rect_norm
yolo_table_roi_min_area_ratio
yolo_table_roi_max_area_ratio
yolo_table_roi_ema_alpha
yolo_table_roi_anchor
yolo_table_roi_lower_ratio
```

## 2. RGB 到 depth 的映射

默认使用 centered_scale：

```text
depth_x_norm = 0.5 + rgb_depth_center_offset_x + (rgb_x_norm - 0.5) * rgb_fov_in_depth_scale_x
depth_y_norm = 0.5 + rgb_depth_center_offset_y + (rgb_y_norm - 0.5) * rgb_fov_in_depth_scale_y
```

当 RGB 视野是 depth 中心 75% 时：

```yaml
rgb_fov_in_depth_scale_x: 0.75
rgb_fov_in_depth_scale_y: 0.75
```

当 RGB 和 depth FOV 完全一致时：

```yaml
rgb_fov_in_depth_scale_x: 1.0
rgb_fov_in_depth_scale_y: 1.0
```

如果中心有偏移，只调 offset。

## 3. ROI 大小

ROI 以映射后的 depth bbox 中心为中心：

```text
roi_w = mapped_bbox_w * yolo_table_roi_scale_x
roi_h = mapped_bbox_h * yolo_table_roi_scale_y
```

小于 1 是缩小，大于 1 是放大。

## 4. YOLO 丢失时的 ROI hold

短时 YOLO 丢失时，继续使用最后一次有效 YOLO 推出的 ROI：

```text
roi_source = yolo_table_bbox_hold
roi_hold_active = true
```

超过 `yolo_table_bbox_hold_frames` 后，不再切到右下角/静态 fallback，而是：

```text
roi_source = disabled_no_table_bbox
```

这样控制层应进入 SEARCH / local_rotate_search，不再让 docking 使用错误 ROI。
