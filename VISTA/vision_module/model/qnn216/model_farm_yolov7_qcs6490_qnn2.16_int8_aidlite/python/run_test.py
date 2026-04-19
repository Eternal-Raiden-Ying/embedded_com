import time
import numpy as np
import cv2
from utils import *
import os
import aidlite
import argparse
import onnxruntime
from yolov7_head import Detect



def main(args):   
    print("Start main ... ...")
    # aidlite.set_log_level(aidlite.LogLevel.INFO)
    # aidlite.log_to_stderr()
    # print(f"Aidlite library version : {aidlite.get_library_version()}")
    # print(f"Aidlite python library version : {aidlite.get_py_library_version()}")
    config = aidlite.Config.create_instance()
    if config is None:
        print("Create config failed !")
        return False
    
    config.implement_type = aidlite.ImplementType.TYPE_LOCAL
    if args.model_type.lower()=="qnn":
        config.framework_type = aidlite.FrameworkType.TYPE_QNN
    elif args.model_type.lower()=="snpe2" or args.model_type.lower()=="snpe":
        config.framework_type = aidlite.FrameworkType.TYPE_SNPE2
        
    config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
    config.is_quantify_model = 1

    model = aidlite.Model.create_instance(args.target_model)
    if model is None:
        print("Create model failed !")
        return False
    size=640
    input_shapes = [[1, size, size, 3]]
    output_shapes = [[1, int(size/8), int(size/8), (args.cls_num+5)*3], [1, int(size/16), int(size/16), (args.cls_num+5)*3], [1, int(size/32), int(size/32), (args.cls_num+5)*3]]
    anchors = [[12,16, 19,36, 40,28],
                [36,75, 76,55, 72,146],
                [142,110, 192,243, 459,401]]
    stride = [8, 16, 32]
    yolov7_head = Detect(args.cls_num, anchors, stride, size)
    
    model.set_model_properties(input_shapes, aidlite.DataType.TYPE_FLOAT32,
                               output_shapes, aidlite.DataType.TYPE_FLOAT32)

    interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
    if interpreter is None:
        print("build_interpretper_from_model_and_config failed !")
        return None
    result = interpreter.init()
    if result != 0:
        print(f"interpreter init failed !")
        return False
    result = interpreter.load_model()
    if result != 0:
        print("interpreter load model failed !")
        return False
    print("detect model load success!")
    
    # image process
    frame = cv2.imread(args.imgs)
    # 图片做等比缩放
    img_input = preprocess_img(frame,target_shape=(size,size),
                                            div_num=255,
                                            means=None,
                                            stds=None)
    
    # qnn run
    invoke_time=[]
    for i in range(args.invoke_nums):
        result = interpreter.set_input_tensor(0, img_input.data)
        if result != 0:
            print("interpreter set_input_tensor() failed")
        
        t1=time.time()
        result = interpreter.invoke()
        cost_time = (time.time()-t1)*1000
        invoke_time.append(cost_time)
        
        if result != 0:
            print("interpreter set_input_tensor() failed")
        
        stride8 = interpreter.get_output_tensor(0)
        stride16 = interpreter.get_output_tensor(1)
        stride32 = interpreter.get_output_tensor(2)

    result = interpreter.destory()
    
    ## time 统计
    max_invoke_time = max(invoke_time)
    min_invoke_time = min(invoke_time)
    mean_invoke_time = sum(invoke_time)/args.invoke_nums
    var_invoketime=np.var(invoke_time)
    print("=======================================")
    print(f"QNN inference {args.invoke_nums} times :\n --mean_invoke_time is {mean_invoke_time} \n --max_invoke_time is {max_invoke_time} \n --min_invoke_time is {min_invoke_time} \n --var_invoketime is {var_invoketime}")
    print("=======================================")
        
    ##  后处理
    # concat qnn_out
    validCount0 = stride8.reshape(*output_shapes[0]).transpose(0, 3, 1, 2)
    validCount1 = stride16.reshape(*output_shapes[1]).transpose(0, 3, 1, 2)
    validCount2 = stride32.reshape(*output_shapes[2]).transpose(0, 3, 1, 2)

    pred = yolov7_head([validCount0, validCount1, validCount2])

    det_pred = detect_postprocess(pred, frame.shape, [size, size, 3], conf_thres=0.5, iou_thres=0.45)
    res_img = draw_detect_res(frame, det_pred)
     
    # 画图
    cv2.imwrite("./python/bus_result.jpg", res_img)   
    print("=======================================")
    
def parser_args():
    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument('--target_model',type=str,default='./models/cutoff_yolov7_w8a8.qnn216.ctx.bin',help="inference model path")
    parser.add_argument('--imgs',type=str,default='./python/bus.jpg',help="Predict images path")
    parser.add_argument('--cls_num',type=int,default=80,help="The number of targets detected")
    parser.add_argument('--invoke_nums',type=int,default=10,help="Inference nums")
    parser.add_argument('--model_type',type=str,default='QNN',help="run backend")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parser_args()
    main(args)

