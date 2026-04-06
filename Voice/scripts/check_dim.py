import numpy as np
import onnxruntime as ort
from openwakeword.utils import AudioFeatures

WAKE_ONNX = "/home/aidlux/2026/Voice/kws/wake_nihao_xiaoche_clean.onnx"

# 1) TFLite 前处理
af = AudioFeatures(inference_framework="tflite")

# 2) ONNX 最终分类器
sess = ort.InferenceSession(WAKE_ONNX, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]

print("wake onnx input :", inp.name, inp.shape, inp.type)
print("wake onnx output:", out.name, out.shape, out.type)

# 这里会得到 [1, T, 96]，例如 [1,16,96] 或 [1,7,96]
T = int(inp.shape[1])
print("need feature frames T =", T)

# 给 4 秒随机音频初始化一次 buffer
dummy = np.random.randint(-1000, 1000, 16000 * 4).astype(np.int16)
af(dummy)

# 模拟喂一帧 80ms 音频
frame = np.zeros(1280, dtype=np.int16)
af(frame)

# 取出特征
feat = af.get_features(T).astype(np.float32)
print("feature shape =", feat.shape)

# 跑最终 ONNX 分类器
pred = sess.run(None, {inp.name: feat})[0]
print("pred =", pred)