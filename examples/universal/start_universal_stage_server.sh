#!/bin/bash
# Universal Stage API Server Startup Script
# Launches an async API server for video processing operators

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "$DIR/.." >/dev/null 2>&1 && pwd )"

cd "$PROJECT_ROOT" || exit

# Set environment variables
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export VLLM_DEVICE_MEMORY_FRACTION=0.5
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Disable model auto-download from HuggingFace
export HF_HUB_OFFLINE=1

echo "Starting Universal Stage API server on port 8091..."
echo "Configuration: $DIR/universal_stage_config.yaml"
echo "Using model: demo"
echo ""

# Start the async omni server
python3 -m vllm_omni.entrypoints.cli.main serve demo \
    --omni \
    --stage-configs-path "$DIR/universal_stage_config.yaml" \
    --port 8091 \
    --disable-log-requests \
    --tensor-parallel-size 1 \
    "$@"


