## 模型信息 (Model Information)
### 原始模型 (Source model)
- 输入尺寸 (Input shape): 640x640
- 参数数量 (Number of parameters): 35.19M
- 模型大小 (Model size): 144.73M
- 输出尺寸 (Output shape): 1x25200x85

原始模型仓库: [yolov7](https://github.com/WongKinYiu/yolov7)

### 转换后的模型 (Converted model)

- 精度 (Precision): INT8
- 后端 (Backend): QNN2.16
- 目标设备 (Target Device): FV01 QCS6490

## 使用 AidLite SDK 进行推理 (Inference with AidLite SDK)

### SDK 安装 (SDK installation)
Model Farm 使用 AidLite SDK 作为模型推理 SDK。详情请参考 [AidLite 开发者文档](https://v2.docs.aidlux.com/en/sdk-api/aidlite-sdk/)

- 安装 AidLite SDK：

```bash
# 安装合适版本的 aidlite sdk
sudo aid-pkg update
sudo aid-pkg install aidlite-sdk
# 下载与上述后端匹配的 qnn 版本。例如安装 QNN2.23 Aidlite: sudo aid-pkg install aidlite-qnn223
sudo aid-pkg install aidlite-{QNN VERSION}
```

- 验证 AidLite SDK：

```bash
# aidlite sdk C++ 检查
python3 -c "import aidlite ; print(aidlite.get_library_version())"

# aidlite sdk Python 检查
python3 -c "import aidlite ; print(aidlite.get_py_library_version())"
```

### 运行 Python Demo (Run python Demo)
```bash
cd yolov7/model_farm_yolov7_qcs6490_qnn2.16_int8_aidlite
python3  python/run_test.py --target_model ./models/cutoff_yolov7_w8a8.qnn216.ctx.bin --imgs ./python/bus.jpg  --invoke_nums 10
```

### 运行 C++ Demo (Run c++ demo)

```bash
cd yolov7/model_farm_yolov7_qcs6490_qnn2.16_int8_aidlite/cpp
mkdir build 
cd build 
cmake ..
make
./run_yolov7
```