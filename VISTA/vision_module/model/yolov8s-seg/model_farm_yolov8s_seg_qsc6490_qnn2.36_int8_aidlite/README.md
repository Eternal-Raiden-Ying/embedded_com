## Model Information
## Source model
- Input shape: 640x640
- Number of parameters: 11.27M
- Model size: 45.22M
- Output shape: 1x32x160x160, 1x116x8400

Source model repository: [yolov8](https://github.com/ultralytics/ultralytics)

### Converted model

- Precision: INT8
- Backend: QNN2.36
- Target Device: FV01 QCS6490

## Model Conversion Reference
User can find model conversion reference at [aimo.aidlux.com](https://aimo.aidlux.com/#/public/5099ed8c-bfd4-4cc9-ad5c-392d75bcaa56)

## Inference with AidLite SDK

### SDK installation
Model Farm uses AidLite SDK as the model inference SDK. For details, please refer to the [AidLite Developer Documentation](https://docs.aidlux.com/guide/software/sdk/aidlite/aidlite-sdk)

- Install AidLite SDK

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

### Run demo
#### python
```bash
cd model_farm_yolov8s_seg_qsc6490_qnn2.36_int8_aidlite
python3 ./python/run_test.py --target_model ./models/cutoff_yolov8s-seg_qcs6490_w8a8.qnn236.ctx.bin --imgs ./python/bus.jpg  --invoke_nums 10
```

#### cpp
```bash
cd model_farm_yolov8s_seg_qsc6490_qnn2.36_int8_aidlite/cpp
mkdir build && cd build 
cmake .. && make
./run_test
```

