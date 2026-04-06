import os
import sys
import cv2
import numpy as np
from utils import *
# AidLux 的底层推理库
import aidlite

# ==========================================
# 1. 仿 Ultralytics 的 Results 封装类
# ==========================================
class YOLOResults:
    def __init__(self, orig_img, boxes, masks, classes, confs, names=None):
        self.orig_img = orig_img
        self.boxes = boxes     # [N, 4] xyxy 格式
        self.masks = masks     # [N, H, W] 对应原图分辨率的二值化掩码
        self.classes = classes # [N] 类别 ID
        self.confs = confs     # [N] 置信度
        self.names = names or {i: f"class_{i}" for i in range(80)}

    def plot(self, conf_thres=0.25):
        """提供原生的可视化方法，自动绘制边界框和半透明掩码"""
        canvas = self.orig_img.copy()
        if len(self.boxes) == 0:
            return canvas

        for box, mask, cls_id, conf in zip(self.boxes, self.masks, self.classes, self.confs):
            if conf < conf_thres:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            color = np.random.RandomState(int(cls_id)).randint(0, 255, size=3).tolist()
            
            # 1. 绘制半透明掩码
            # 将掩码阈值化并转为布尔索引
            m_bool = mask > 0.5 
            canvas[m_bool] = canvas[m_bool] * 0.5 + np.array(color) * 0.5
            
            # 2. 绘制边界框与标签
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{self.names.get(int(cls_id), str(cls_id))} {conf:.2f}"
            
            # 标签底色块
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(canvas, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return canvas

# ==========================================
# 2. QNN YOLO26 实例分割主类
# ==========================================
class QNNYOLO26Seg:
    def __init__(self, model_path, conf=0.25, iou=0.45, imgsz=640, sdk_version="2.36"):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        
        # 1. 初始化 Aidlite Config
        aidlite.set_log_level(aidlite.LogLevel.INFO)
        config = aidlite.Config.create_instance()
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1                             
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        
        # ⚠️ 注意：根据你的 qnn_info，SDK 版本是 2.36，所以这里用 TYPE_QNN236 或者通用的 TYPE_QNN
        config.framework_type = aidlite.FrameworkType.TYPE_QNN

        # 2. 初始化 Model 属性
        model = aidlite.Model.create_instance(model_path)
        
        # ==========================================
        # 🚀 核心修改 1：重写输出形状 (与 qnn_info 完美对齐)
        # ==========================================
        input_shapes = [[1, imgsz, imgsz, 3]]
        
        # 注意顺序：AidLux 获取输出张量时，通常是按字母顺序或者内部算子图的顺序。
        # 安全起见，我们定义 4 个输出形状。稍后在 forward 里我们会根据 shape 动态辨别它们。
        output_shapes = [
            [1, 32, 8400],       # mask_coefs
            [1, 4, 8400],        # bboxes
            [1, 20, 8400],       # scores
            [1, 160, 160, 32]    # mask_protos
        ]
        
        model.set_model_properties(
            input_shapes, aidlite.DataType.TYPE_FLOAT32,
            output_shapes, aidlite.DataType.TYPE_FLOAT32
        )

        # 3. 构建并加载 Interpreter
        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        if self.interpreter.init() != 0:
            raise RuntimeError("Interpreter init failed!")
        if self.interpreter.load_model() != 0:
            raise RuntimeError("Interpreter load model failed!")
        print("✅ AidLux QNN Interpreter Load Success! (Output Splitting Mode)")
    
    @staticmethod
    def _eqprocess(image, size1, size2):
        h, w, _ = image.shape
        mask = np.zeros((size1, size2, 3), dtype=np.float32)
        scale1 = h / size1
        scale2 = w / size2
        if scale1 > scale2:
            scale = scale1
        else:
            scale = scale2
        img = cv2.resize(image, (int(w / scale), int(h / scale)))
        mask[: int(h / scale), : int(w / scale), :] = img
        return mask, scale
        
    @staticmethod
    def letterbox(
        im,
        new_shape,
        color=(114, 114, 114),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        """
        Resize and pad image while meeting stride-multiple constraints
        Returns:
            im (array): (height, width, 3)
            ratio (array): [w_ratio, h_ratio]
            (dw, dh) (array): [w_padding h_padding]
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):  # [h_rect, w_rect]
            new_shape = (new_shape, new_shape)
     
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
     
        # Compute padding
        ratio = r, r  # wh ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w h
        dw, dh = (
            new_shape[1] - new_unpad[0],
            new_shape[0] - new_unpad[1],
        )  # wh padding
     
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])  # [w h]
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # [w_ratio, h_ratio]
     
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return im, ratio, (dw, dh)

    def preprocess(self, image):
        # 1. Resize & Pad
        input_img, ratio, pad = self.letterbox(image, (self.imgsz, self.imgsz))
        
        # 2. BGR 转 RGB (极其重要，与网页端量化对齐！)
        input_img = input_img[..., ::-1]
        
        # 3. 扩充 Batch 维度并转为浮点数
        input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
        
        # 4. 强制内存连续
        input_img = np.ascontiguousarray(input_img)
        
        # 5. 🌟 真正的归一化 (而且确保没有被覆盖！)
        blob = input_img / 255.0
        
        # 6. 构建 ratio_pad
        ratio_pad = (ratio[0], pad)
        
        return blob, ratio_pad

    def forward(self, input_tensor):
        """执行推理并获取张量"""
        self.interpreter.set_input_tensor("images", input_tensor) 
        self.interpreter.invoke()
        
        # ==========================================
        # 🚀 核心修改 2：动态获取 4 个张量并拼装
        # ==========================================
        bboxes, scores, mask_coefs, mask_protos = None, None, None, None
        
        # 遍历获取 4 个输出张量，并根据 shape 物理意义自动对号入座
        for i in range(4):
            tensor = self.interpreter.get_output_tensor(i)
            # AidLux 吐出来的可能是 1D 数组，我们需要根据数量辨认
            size = tensor.size
            
            if size == 4 * 8400:          # 33600
                bboxes = tensor.reshape(1, 4, 8400)
            elif size == 80 * 8400:       # 672000
                scores = tensor.reshape(1, 80, 8400)
            elif size == 32 * 8400:       # 268800
                mask_coefs = tensor.reshape(1, 32, 8400)
            elif size == 160 * 160 * 32:  # 819200
                mask_protos = tensor.reshape(1, 160, 160, 32)
                
        # 🛡️ 异常拦截：确保 4 个张量都拿到了
        if any(t is None for t in [bboxes, scores, mask_coefs, mask_protos]):
            raise ValueError("未能获取到所有 4 个拆分张量，请检查 QNN 模型输出配置！")
            
        # ⚡ 终极魔法：在 CPU 上瞬间拼装回 [1, 116, 8400]
        # 严格按照 YOLOv8 原版输出的内部顺序：(bboxes, scores, mask_coefs)
        out0_combined = np.concatenate([bboxes, scores, mask_coefs], axis=1)
        
        # 返回拼装好的 out0，以及独立的原型掩码 out1
        return out0_combined, mask_protos

    def postprocess(self, preds, proto, orig_img, ratio_pad):
        orig_shape = orig_img.shape[:2]
        
        # 1. 纯 Numpy NMS (这里不需要动，因为它依然接收 [1, 116, 8400] 的 preds)
        det = non_max_suppression(preds, self.conf, self.iou, max_det=300)
        
        if len(det) == 0:
            return YOLOResults(orig_img, [], [], [], [])

        # ==========================================
        # 🚀 核心修改 3：修复掩码生成的通道顺序
        # ==========================================
        # 由于 AidLux/QNN 输出的原型掩码是 NHWC [1, 160, 160, 32]
        # 这里需要正确地把它 transpose 回 NCHW [32, 160, 160] 喂给 utils
        proto_chw = proto[0].transpose(2, 0, 1)  
        
        masks = process_mask(proto_chw, det[:, 6:], det[:, :4], (self.imgsz, self.imgsz), upsample=True)

        # 3. 映射回原图尺寸
        det[:, :4] = scale_boxes((self.imgsz, self.imgsz), det[:, :4], orig_shape, ratio_pad).round()
        masks = scale_masks(masks, orig_shape, ratio_pad)

        return YOLOResults(
            orig_img=orig_img,
            boxes=det[:, :4],
            masks=masks,
            classes=det[:, 5],
            confs=det[:, 4]
        )

# ==========================================
# 3. 视频流推理与调用示例
# ==========================================
def stream():
    model_path = "/home/aidlux/yolov5/data/qnn_yolov5_multi/yolo26s-seg_qcs6490_w8a8.qnn240.ctx.bin.amf"
    video_path = "test.mp4"
    out_path = "output.mp4"

    # 初始化预测器
    predictor = QNNYOLO26Seg(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频写出器
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 预处理
        input_tensor, ratio_pad = predictor.preprocess(frame)
        
        # 2. 推理计算
        out0, out1 = predictor.forward(input_tensor)
        
        # 3. 后处理解码
        results = predictor.postprocess(out0, out1, frame, ratio_pad)
        
        # 4. 可视化渲染
        res_frame = results.plot(conf_thres=0.3)
        
        # 写入并显示
        writer.write(res_frame)
        cv2.imshow('AidLux QNN YOLO26-Seg', res_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
def main():
    model_path = "../yolo26s-seg-grasp/yolo26s-seg-grasp_split_qcs6490_w8a8.qnn236.ctx.bin.amf"
    img_pth = "/home/aidlux/2026/VISTA/vision_module/data/test_easy.jpg"
    out_path = "res.jpg"
    
    # 初始化预测器
    predictor = QNNYOLO26Seg(model_path)

    # 1. 读取单张图片
    im0 = cv2.imread(img_pth) # 修正变量名: img_pth
    if im0 is None:
        print(f"读取图片失败，请检查路径: {img_pth}")
        return

    # 2. 预处理计算 (获取处理后的 Tensor 和缩放参数)
    input_tensor, ratio_pad = predictor.preprocess(im0)
    print(f"input tensor shape {input_tensor.shape}")
    
    # 3. 推理计算
    out0, out1 = predictor.forward(input_tensor)
    
    # ==================== 🔍 诊断探针开始 ====================
    print("\n--- 🔍 模型输出诊断 ---")
    print(f"1. out0 shape: {out0.shape}")
    
    # 获取 8400 个预测点中，最大的那个类别置信度
    max_score = np.max(out0[0, 4:84, :])
    print(f"2. ⭐ 当前画面预测出的最高置信度: {max_score:.4f}")
    
    # 检查输入张量的取值范围
    print(f"3. 预处理后的输入范围: min={np.min(input_tensor):.4f}, max={np.max(input_tensor):.4f}")
    print("------------------------\n")
    # ==================== 🔍 诊断探针结束 ====================

    # 4. 后处理解码
    results = predictor.postprocess(out0, out1, im0, ratio_pad)
    print(f"5. NMS过滤后，最终剩余目标数: {len(results.boxes)}")
    
    # 5. 可视化渲染
    res_frame = results.plot(conf_thres=0.25)
    
    # 写入并显示图片
    success = cv2.imwrite(out_path, res_frame) 
    if success:
        print(f"🎉 推理大功告成！结果已保存至: {out_path}")
    else:
        print("❌ 保存图片失败，请检查路径是否有写入权限。")

if __name__ == "__main__":
    main()