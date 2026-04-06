import numpy as np
import cv2
import aidlite
from utils import eqprocess, xywh2xyxy, NMS, process_mask, masks2segments, draw_detect_res
import os
import time
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))

class qnn_predict(object):
    def __init__(self,args) -> None:
        # aidlite.set_log_level(aidlite.LogLevel.INFO)
        # aidlite.log_to_stderr()
        # print(f"Aidlite library version : {aidlite.get_library_version()}")
        # print(f"Aidlite python library version : {aidlite.get_py_library_version()}")
        config = aidlite.Config.create_instance()
        if config is None:
            print("Create model failed !")
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        config.framework_type = aidlite.FrameworkType.TYPE_QNN
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1

        model = aidlite.Model.create_instance(args.target_model)
        if model is None:
            print("Create model failed !")

        self.conf = args.conf_thres
        self.iou=args.iou_thres
        self.width = args.width
        self.height = args.height
        self.class_num = args.class_num
        self.input_shape = [[1,self.height,self.width,3]]
        self.blocks = int(self.height * self.width * ( 1 / 64 + 1 / 256 + 1 / 1024))
        self.maskw = int(self.width / 4)
        self.maskh = int(self.height / 4)
        self.output_shape = [[1,32,self.blocks],[1,4,self.blocks],[1,self.class_num,self.blocks],[1,self.maskh, self.maskw,32]]
        
        model.set_model_properties(self.input_shape, aidlite.DataType.TYPE_FLOAT32, self.output_shape, aidlite.DataType.TYPE_FLOAT32)
        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        if self.interpreter is None:
            print("build_interpretper_from_model_and_config failed !")
        result = self.interpreter.init()
        if result != 0:
            print(f"interpreter init failed !")
        result = self.interpreter.load_model()
        if result != 0:
            print("interpreter load model failed !")
        print("detect model load success!")
        
    def __del__(self):
        self.interpreter.destory()

    def pretreat_img(self,frame): 
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, scale = eqprocess(img, self.height, self.width)
        img = img / 255
        img = img.astype(np.float32)
        return img,scale
    
    def qnn_run(self, orig_imgs,args):
        input_img_f,scale=self.pretreat_img(orig_imgs)  # 图片resize HWC
        input_img = np.expand_dims(input_img_f, 0)
      
        invoke_time=[]
        for i in range(args.invoke_nums):
            self.interpreter.set_input_tensor(0, input_img.data)
            t0 = time.time()
            self.interpreter.invoke()
            t1 = time.time()
            cost_time=(t1-t0)*1000
            invoke_time.append(cost_time)

            input0_data = self.interpreter.get_output_tensor(2).reshape(1,4,self.blocks)
            input1_data = self.interpreter.get_output_tensor(3).reshape(1,self.class_num,self.blocks)
            input2_data = self.interpreter.get_output_tensor(1).reshape(1,32,self.blocks)
            protos = self.interpreter.get_output_tensor(0).reshape(1,self.maskh, self.maskw,32).transpose(0,3,1,2)

            boxes = np.concatenate([input0_data, input1_data, input2_data], axis = 1)
            x = boxes.transpose(0,2,1)
            x = x[np.amax(x[..., 4:-32], axis=-1) > self.conf]
            if len(x) < 1:
                return None, None
        
            x = np.c_[x[..., :4], np.amax(x[..., 4:-32], axis=-1), np.argmax(x[..., 4:-32], axis=-1), x[..., -32:]]
            
            x[:, :4] = xywh2xyxy(x[:, :4])
            index = NMS(x[:, :4], x[:, 4], self.iou)
            out_boxes = x[index]
            out_boxes[..., :4] = out_boxes[..., :4] * scale
            
            masks = process_mask(protos[0], out_boxes[:, -32:], out_boxes[:, :4], orig_imgs.shape)
            segments = masks2segments(masks)
           
        ## time 统计
        max_invoke_time = max(invoke_time)
        min_invoke_time = min(invoke_time)
        mean_invoke_time = sum(invoke_time)/args.invoke_nums
        var_invoketime=np.var(invoke_time)
        print("========================================")
        print(f"QNN inference {args.invoke_nums} times :\n --mean_invoke_time is {mean_invoke_time} \n --max_invoke_time is {max_invoke_time} \n --min_invoke_time is {min_invoke_time} \n --var_invoketime is {var_invoketime}")
        print("========================================")

        return out_boxes, segments
       

def parser_args():
    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument('--target_model',type=str,default='./models/cutoff_yolov8s-seg_qcs6490_w8a8.qnn236.ctx.bin',help="inference model path")
    parser.add_argument('--imgs',type=str,default='./python/bus.jpg',help="Predict images path")
    parser.add_argument('--invoke_nums',type=int,default=10,help="Inference nums")
    parser.add_argument('--model_type',type=str,default='QNN',help="run backend")
    parser.add_argument('--width',type=int,default=640,help="Model input size")
    parser.add_argument('--height',type=int,default=640,help="Model input size")
    parser.add_argument('--conf_thres',type=float,default=0.45,help="confidence threshold for filtering the annotations")
    parser.add_argument('--iou_thres',type=float,default=0.45,help="Iou threshold for filtering the annotations")
    parser.add_argument('--class_num',type=int,default=80,help="Iou threshold for filtering the annotations")
    args = parser.parse_args()
    return args

def main(args):
    model = qnn_predict(args)
    frame = cv2.imread(args.imgs)
    qnn_out_boxes, qnn_segments = model.qnn_run(frame,args)
    print(f"{len(qnn_out_boxes)} targets have been detected.")
    result = draw_detect_res(frame, qnn_out_boxes, qnn_segments)
    cv2.imwrite(f"{current_dir}/bus_result.jpg", result)

if __name__ == "__main__":
    args = parser_args()
    main(args)


