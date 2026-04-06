import numpy as np
import cv2

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def NMS_fast(boxes, scores, thresh):
    """
    ⚡ 调用 OpenCV 底层的 C++ NMS，比纯 Python while 循环快数十倍
    """
    if len(boxes) == 0:
        return []
    
    # cv2.dnn.NMSBoxes 需要 [x, y, w, h] 格式
    bboxes_xywh = np.copy(boxes)
    bboxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    bboxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    
    indices = cv2.dnn.NMSBoxes(bboxes_xywh.tolist(), scores.tolist(), 0.0, thresh)
    if len(indices) > 0:
        return indices.flatten()
    return []

def crop_mask(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask_fast(protos, masks_in, bboxes, input_shape=(640, 640)):
    """
    ⚡ 针对硬件直出 640x640 优化的极简、极速版掩码处理
    省去了 UMat 拷贝开销和黑边计算，直接放大！
    """
    if len(masks_in) == 0:
        return np.zeros((0, input_shape[0], input_shape[1]), dtype=bool)

    # 1. (N, 32) 乘 (32, 25600) = (N, 25600)
    c, mh, mw = protos.shape
    masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw))
    
    # 2. 直接在 CPU 用 cv2 拉伸到 640x640 (比来回倒腾 GPU 更快)
    masks = masks.transpose(1, 2, 0) # (160, 160, N)
    masks = cv2.resize(masks, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
        
    masks = masks.transpose(2, 0, 1) # (N, 640, 640)

    # 3. 边界裁剪并二值化输出
    masks = crop_mask(masks, bboxes)
    return np.greater(masks, 0.5)

