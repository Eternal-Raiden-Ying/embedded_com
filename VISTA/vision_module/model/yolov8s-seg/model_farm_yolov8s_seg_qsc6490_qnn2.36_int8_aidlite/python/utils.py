import numpy as np
import cv2

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")
           
def eqprocess(image, size1, size2):
    h,w,_ = image.shape
    mask = np.zeros((size1,size2,3),dtype=np.float32)
    scale1 = h /size1
    scale2 = w / size2
    if scale1 > scale2:
        scale = scale1
    else:
        scale = scale2
    img = cv2.resize(image,(int(w / scale),int(h / scale)))
    mask[:int(h / scale),:int(w / scale),:] = img
    return mask, scale

def xywh2xyxy(x):
    '''
    Box (center x, center y, width, height) to (x1, y1, x2, y2)
    '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(box):
    '''
    Box (left_top x, left_top y, right_bottom x, right_bottom y) to (left_top x, left_top y, width, height)
    '''
    box[:, 2:] = box[:, 2:] - box[:, :2]
    return box

def NMS(dets, scores, thresh):
    '''
    单类NMS算法
    dets.shape = (N, 5), (left_top x, left_top y, right_bottom x, right_bottom y, Scores)
    '''
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    areas = (y2-y1+1) * (x2-x1+1)
    keep = []
    index = scores.argsort()[::-1]
    while index.size >0:
        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22-x11+1)    # the weights of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap
        overlaps = w*h
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
        idx = np.where(ious<=thresh)[0]
        index = index[idx+1]   # because index start from 1
 
    return keep

def draw_detect_res(img, det_pred, segments):
    '''
    检测结果绘制
    '''
    if det_pred is None:
        return img

    img = img.astype(np.uint8)
    im_canvas = img.copy()
    color_step = int(255/len(CLASSES))
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        cls_id = int(det_pred[i][5])
        cv2.putText(img, f'{CLASSES[cls_id]}', (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, int(cls_id*color_step), int(255-cls_id*color_step)),thickness = 2)
        if len(segments[i]) > 0:
            cv2.polylines(img, np.int32([segments[i]]), True, (0, int(cls_id*color_step), int(255-cls_id*color_step)), 2)
            cv2.fillPoly(img, np.int32([segments[i]]), (0, int(cls_id*color_step), int(255-cls_id*color_step)))
    img = cv2.addWeighted(im_canvas, 0.3, img, 0.7, 0)
    return img

def scale_mask(masks, im0_shape):
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks

def crop_mask(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, im0_shape):
    c, mh, mw = protos.shape
    masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
    masks = np.ascontiguousarray(masks)
    masks = scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
    masks = np.einsum('HWN -> NHW', masks)  # HWN -> NHW
    masks = crop_mask(masks, bboxes)
    return np.greater(masks, 0.5)

def masks2segments(masks):
    segments = []
    for x in masks.astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
        if c:
            c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments
