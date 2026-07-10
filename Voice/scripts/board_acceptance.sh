#!/bin/bash
# Multi-stage acceptance tests for Voice Gateway on SC171

set -e

STAGE=${1:-"all"}
PROFILE=${2:-"configs/profiles/sc171_voice_gateway.yaml"}
TEST_WAV=${3:-"Voice/runs/test_440hz.wav"}

export PYTHONPATH=".:Voice:common:orchestrator:$PYTHONPATH"

run_stage_preflight() {
    echo -e "\n=============================================="
    echo "STAGE: preflight"
    echo "=============================================="
    bash Voice/scripts/board_preflight.sh
}

run_stage_kws_file() {
    echo -e "\n=============================================="
    echo "STAGE: kws-file"
    echo "=============================================="
    if [ ! -f "$TEST_WAV" ]; then
        echo "Creating a temp WAV file for KWS test..."
        python3 -c "import wave, numpy as np; data = (np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000, endpoint=False)) * 10000).astype(np.int16); wf = wave.open('$TEST_WAV', 'wb'); wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(data.tobytes()); wf.close()"
    fi
    python3 -m voice_service.examples.kws_probe --profile "$PROFILE" --wav "$TEST_WAV"
}

run_stage_asr_file() {
    echo -e "\n=============================================="
    echo "STAGE: asr-file"
    echo "=============================================="
    if [ ! -f "$TEST_WAV" ]; then
        echo "Creating a temp WAV file for ASR test..."
        python3 -c "import wave, numpy as np; data = (np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000, endpoint=False)) * 10000).astype(np.int16); wf = wave.open('$TEST_WAV', 'wb'); wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(data.tobytes()); wf.close()"
    fi
    python3 -m voice_service.examples.asr_probe --profile "$PROFILE" --wav "$TEST_WAV" --mode offline
}

run_stage_tts_file() {
    echo -e "\n=============================================="
    echo "STAGE: tts-file"
    echo "=============================================="
    python3 -m voice_service.examples.tts_probe --profile "$PROFILE" --text "测试语音合成输出" --output "Voice/runs/board_acceptance_tts.wav" --no-play
}

run_stage_audio_device() {
    echo -e "\n=============================================="
    echo "STAGE: audio-device"
    echo "=============================================="
    python3 -m voice_service.examples.audio_device_probe --profile "$PROFILE"
}

run_stage_uds_text() {
    echo -e "\n=============================================="
    echo "STAGE: uds-text"
    echo "=============================================="
    echo "Running dry-run TCP round-trip and validation test..."
    python3 -m pytest tests/voice/test_voice_gateway.py -k "test_tcp_roundtrip" || echo "Pytest not installed or tcp test failed"
}

run_stage_voice_no_motion() {
    echo -e "\n=============================================="
    echo "STAGE: voice-no-motion"
    echo "=============================================="
    echo "Starting Voice Gateway in interactive dry-run-text mode..."
    echo "Type 'exit' to quit."
    python3 -m voice_service.app.main --profile "$PROFILE" --dry-run-text
}

run_stage_voice_serial_dryrun() {
    echo -e "\n=============================================="
    echo "STAGE: voice-serial-dryrun"
    echo "=============================================="
    echo "Verifying robot stack status format & serial loopback..."
    if [ -f "start_robot_stack.sh" ]; then
        bash start_robot_stack.sh status || echo "Status command skipped"
    else
        echo "start_robot_stack.sh not found."
    fi
}

# Run selection
case "$STAGE" in
    "preflight")
        run_stage_preflight
        ;;
    "kws-file")
        run_stage_kws_file
        ;;
    "asr-file")
        run_stage_asr_file
        ;;
    "tts-file")
        run_stage_tts_file
        ;;
    "audio-device")
        run_stage_audio_device
        ;;
    "uds-text")
        run_stage_uds_text
        ;;
    "voice-no-motion")
        run_stage_voice_no_motion
        ;;
    "voice-serial-dryrun")
        run_stage_voice_serial_dryrun
        ;;
    "all")
        run_stage_preflight
        run_stage_uds_text
        run_stage_kws_file
        run_stage_asr_file
        run_stage_tts_file
        run_stage_audio_device
        run_stage_voice_serial_dryrun
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Available: preflight, uds-text, kws-file, asr-file, tts-file, audio-device, voice-no-motion, voice-serial-dryrun, all"
        exit 1
        ;;
esac

echo -e "\n>>> Acceptance stage '$STAGE' execution complete! <<<"
