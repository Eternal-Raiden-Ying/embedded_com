#########################################
#              yolo target              #
#########################################

coco80 = (
    "person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
    "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
    "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop", "mouse", "remote ", "keyboard ", "cell phone", "microwave ",
    "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush "
)


grasping_coco20 = (
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "orange",
    "broccoli",
    "carrot",
    "mouse",
    "remote",
    "cell phone",
    "book",
    "clock",
    "scissors",
    "teddy bear",
    "toothbrush"
    )








#########################################
#            asr vocabulary             #
#########################################

asr_class_map = {
    "cup": {"cup"},
    "bottle": {"bottle"},
    "phone": {"cell phone"},
    "remote": {"remote"},
    "apple": {"apple"},
    "banana": {"banana"},
    "book": {"book"},
    # 以下目标当前模型里没有可靠类别，先保留接口，返回 found=false
    "medicine_box": set(),
    "keys": set(),
    "wallet": set(),
    
    "mouse":{"mouse"},  # 暂时用于测试
}









TARGET_CLASSES, ASR_VOCAB_MAP = grasping_coco20, asr_class_map