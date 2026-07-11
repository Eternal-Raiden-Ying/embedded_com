#!/bin/bash
# Board Preflight check script for Voice Gateway on SC171

set -e

echo "=== SC171 Voice Gateway Preflight Checks ==="

# 1. System Info
echo "System date: $(date)"
echo "Kernel info: $(uname -a)"
echo "CPU Architecture: $(uname -m)"

# 2. Disk Space Checks
echo -e "\n--- Disk Space Status ---"
df -h .

# 3. Directory Permissions
echo -e "\n--- Folder Permissions ---"
ls -ld Voice/
ls -ld Voice/config/
ls -ld Voice/ONNX/ 2>/dev/null || echo "ONNX dir not found"
ls -ld Voice/kws/ 2>/dev/null || echo "KWS dir not found"
ls -ld Voice/tts/ 2>/dev/null || echo "TTS dir not found"

# 4. Python Environment & Library Imports Check
echo -e "\n--- Python & Package Import Verification ---"
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "ERROR: python3 not found in PATH!"
    exit 1
fi

echo "Python version: $($PYTHON_CMD --version)"

PACKAGES=("numpy" "msgpack" "yaml" "onnxruntime")
for pkg in "${PACKAGES[@]}"; do
    if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        echo "  [PASS] Import $pkg succeeded"
    else
        echo "  [WARN] Import $pkg failed! Certain components may run in dry-run/mock mode only."
    fi
done

# Check board-specific libraries
BOARD_LIBS=("funasr_onnx" "piper")
for pkg in "${BOARD_LIBS[@]}"; do
    if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        echo "  [PASS] Import $pkg (board runtime) succeeded"
    else
        echo "  [INFO] Import $pkg missing. This is normal on Windows/dev hosts, but required for board execution."
    fi
done

# 5. Model File Validation
echo -e "\n--- Model File Existence ---"
MODELS=(
    "Voice/kws/wake_nihao_xiaoche_v2.onnx"
    "Voice/kws/stop_smallcar_v1.onnx"
    "Voice/ONNX/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx/model_quant.onnx"
    "Voice/ONNX/speech_fsmn_vad_zh-cn-16k-common-onnx/model_quant.onnx"
    "Voice/tts/zh_CN-huayan-x_low/zh_CN-huayan-x_low.onnx"
    "Voice/tts/zh_CN-huayan-x_low/zh_CN-huayan-x_low.onnx.json"
)

ALL_MODELS_FOUND=true
for m in "${MODELS[@]}"; do
    if [ -f "$m" ]; then
        echo "  [FOUND] $m ($(du -sh "$m" | cut -f1))"
    else
        echo "  [MISSING] $m"
        ALL_MODELS_FOUND=false
    fi
done

if [ "$ALL_MODELS_FOUND" = true ]; then
    echo -e "\n[PREFLIGHT SUCCESS] All required board models found."
else
    echo -e "\n[PREFLIGHT WARNING] Some board models are missing! Check paths or manifests."
fi
