import numpy as np
import cv2

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nc=80, nm=32):
    """纯 Numpy 版本的 NMS，利用 OpenCV 加速"""
    # 1. 整理维度: (1, 116, 8400) -> (8400, 116)
    pred = np.squeeze(prediction).T
    
    # 2. 提取并分割数据
    boxes = pred[:, :4]           # 框坐标 (cx, cy, w, h)
    scores = pred[:, 4:4+nc]      # 类别概率 (80个)
    mask_coeffs = pred[:, 4+nc:]  # 掩码系数 (32个)

    # 3. 寻找每个框的最大置信度及其对应的类别 ID
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    # 4. 根据阈值进行初步过滤 (极大提升 NMS 速度)
    valid_mask = max_scores > conf_thres
    boxes = boxes[valid_mask]
    max_scores = max_scores[valid_mask]
    class_ids = class_ids[valid_mask]
    mask_coeffs = mask_coeffs[valid_mask]

    if len(boxes) == 0:
        return np.zeros((0, 6 + nm))

    # 5. 坐标转换: (cx, cy, w, h) -> (x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 6. 多类别 NMS 偏置策略 (避免不同类别的框互相被抑制)
    max_wh = 7680
    boxes_offset = boxes_xyxy + (class_ids * max_wh)[:, None]

    # OpenCV 的 NMS 接受 [x_top_left, y_top_left, width, height] 格式
    boxes_cv2 = np.stack([
        boxes_offset[:, 0], 
        boxes_offset[:, 1], 
        boxes_offset[:, 2] - boxes_offset[:, 0], 
        boxes_offset[:, 3] - boxes_offset[:, 1]
    ], axis=1)

    # 7. 调用 OpenCV C++ 底层 NMS 算法
    indices = cv2.dnn.NMSBoxes(boxes_cv2.tolist(), max_scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return np.zeros((0, 6 + nm))
        
    indices = indices.flatten()[:max_det]

    # 8. 拼合最终结果 [x1, y1, x2, y2, conf, cls, mask_coeffs...]
    final_boxes = boxes_xyxy[indices]
    final_scores = max_scores[indices, None]
    final_class_ids = class_ids[indices, None].astype(np.float32)
    final_masks = mask_coeffs[indices]

    return np.concatenate([final_boxes, final_scores, final_class_ids, final_masks], axis=1)

def crop_mask(masks, boxes):
    """Numpy 掩码裁剪：将检测框外部的掩码区域清零"""
    n, h, w = masks.shape
    boxes = np.round(boxes).astype(int)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        masks[i, :max(0, y1), :] = 0
        masks[i, min(h, y2):, :] = 0
        masks[i, :, :max(0, x1)] = 0
        masks[i, :, min(w, x2):] = 0
    return masks

def process_mask(protos, masks_in, bboxes, shape, upsample=True):
    """Numpy 掩码合成：系数矩阵运算并缩放裁剪"""
    c, mh, mw = protos.shape  # (32, 160, 160)
    
    # 矩阵乘法：(N, 32) @ (32, 160*160) -> (N, 160, 160)
    protos_flat = protos.reshape(c, -1)
    masks = (masks_in @ protos_flat).reshape(-1, mh, mw)

    # 将 640x640 尺度的边界框缩放到 160x160 以匹配原型掩码尺寸
    width_ratio, height_ratio = mw / shape[1], mh / shape[0]
    downsampled_bboxes = bboxes * np.array([width_ratio, height_ratio, width_ratio, height_ratio])
    
    # 裁剪框外多余掩码
    masks = crop_mask(masks, downsampled_bboxes)

    # 恢复原图大小
    if upsample:
        upsampled_masks = np.zeros((masks.shape[0], shape[0], shape[1]), dtype=np.float32)
        for i in range(masks.shape[0]):
            upsampled_masks[i] = cv2.resize(masks[i], (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        masks = upsampled_masks

    # Ultralytics 的逻辑是在插值后使用 gt_(0.0)，因为 sigmoid(x)>0.5 等价于 x>0
    return (masks > 0.0).astype(np.uint8)

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad):
    """Numpy 边界框还原缩放"""
    gain = ratio_pad[0]
    pad_x, pad_y = ratio_pad[1]

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= gain

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img0_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img0_shape[0])
    return boxes

def scale_masks(masks, img0_shape, ratio_pad):
    """Numpy 掩码还原缩放"""
    im1_h, im1_w = masks.shape[1:]
    im0_h, im0_w = img0_shape[:2]

    gain = ratio_pad[0]
    pad_w, pad_h = ratio_pad[1]

    top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
    bottom = int(im1_h - round(pad_h + 0.1))
    right = int(im1_w - round(pad_w + 0.1))

    # 去掉 Letterbox 的黑边
    cropped_masks = masks[:, top:bottom, left:right]

    # 将剥离黑边后的有效区域放大回原始相机分辨率
    scaled_masks = np.zeros((masks.shape[0], im0_h, im0_w), dtype=np.uint8)
    for i in range(masks.shape[0]):
        # 对 boolean/uint8 掩码使用最近邻插值是最快且安全的
        scaled_masks[i] = cv2.resize(cropped_masks[i], (im0_w, im0_h), interpolation=cv2.INTER_NEAREST)

    return scaled_masks