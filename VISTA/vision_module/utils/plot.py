import aidcv as cv2
import numpy as np

from ..config.data import grasping_coco20 as CLASSES

# 预先生成全局颜色表，极大地加快画图速度
COLORS = {i: (0, int(i*(255/len(CLASSES))), int(255-i*(255/len(CLASSES)))) for i in range(len(CLASSES))}

def draw_detect_res_fast(img_bgr, det_pred, masks):
    """
    ⚡ 纯 NumPy 矩阵切片级渲染掩码，完全舍弃低效的 fillPoly 多边形轮廓
    """
    if det_pred is None or len(det_pred) == 0:
        return img_bgr

    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        cls_id = int(det_pred[i][5])
        color = COLORS.get(cls_id, (0, 255, 0))

        # 掩码像素级半透明渲染 (底层 C 语言级别提速)
        m = masks[i]
        img_bgr[m] = img_bgr[m] * 0.5 + np.array(color) * 0.5
        
        # 绘制边界框
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(img_bgr, f'{CLASSES[cls_id]} {det_pred[i][4]:.2f}', (x1, y1-6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_bgr