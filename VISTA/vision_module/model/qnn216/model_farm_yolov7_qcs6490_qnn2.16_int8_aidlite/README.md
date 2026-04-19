## Model Information
### Source model
- Input shape: 640x640
- Number of parameters: 35.19M
- Model size: 144.73M
- Output shape: 1x25200x85

Source model repository: [yolov7](https://github.com/WongKinYiu/yolov7)

### Converted model

- Precision: INT8
- Backend: QNN2.16
- Target Device: FV01 QCS6490

## Inference with AidLite SDK

### SDK installation
Model Farm uses AidLite SDK as the model inference SDK. For details, please refer to the [AidLite Developer Documentation](https://v2.docs.aidlux.com/en/sdk-api/aidlite-sdk/)

- install AidLite SDK

```bash
# Install the appropriate version of the aidlite sdk
sudo aid-pkg update
sudo aid-pkg install aidlite-sdk
# Download the qnn version that matches the above backend. Eg Install QNN2.23 Aidlite: sudo aid-pkg install aidlite-qnn223
sudo aid-pkg install aidlite-{QNN VERSION}
```

- Verify AidLite SDK

```bash
# aidlite sdk c++ check
python3 -c "import aidlite ; print(aidlite.get_library_version())"

# aidlite sdk python check
python3 -c "import aidlite ; print(aidlite.get_py_library_version())"
```

### Run python Demo
```bash
cd yolov7/model_farm_yolov7_qcs6490_qnn2.16_int8_aidlite
python3  python/run_test.py --target_model ./models/cutoff_yolov7_w8a8.qnn216.ctx.bin --imgs ./python/bus.jpg  --invoke_nums 10
```

### Run c++ demo

```bash
cd yolov7/model_farm_yolov7_qcs6490_qnn2.16_int8_aidlite/cpp
mkdir build 
cd build 
cmake ..
make
./run_yolov7
```