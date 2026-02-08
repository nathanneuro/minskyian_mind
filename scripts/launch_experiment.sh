#!/bin/bash
# Launch Minsky Society of Mind experiment
# Server: 2x 4090 GPUs, 500GB RAM

# Set CUDA environment for RWKV JIT compilation
# Try common CUDA installation paths
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif [ -d "/usr/lib/cuda" ]; then
    export CUDA_HOME="/usr/lib/cuda"
elif [ -n "$CUDA_PATH" ]; then
    export CUDA_HOME="$CUDA_PATH"
fi

# Also ensure CUDA bin and lib are in PATH
if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "CUDA_HOME set to: $CUDA_HOME"
else
    echo "WARNING: Could not find CUDA installation. RWKV JIT may fail."
    echo "Set CUDA_HOME manually if needed."
fi

echo "====================================================================="
echo "Minsky Society of Mind - Experiment Launch"
echo "====================================================================="

# Create directories
mkdir -p logs data outputs models

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/experiment_${TIMESTAMP}.log"

# Check if RWKV model exists, download both models if needed
MODEL_DIR="data/models"
if ! ls "$MODEL_DIR"/*.pth 1> /dev/null 2>&1; then
    echo ""
    echo "Downloading models (RWKV 7.2B + T5Gemma)..."
    uv run python scripts/download_model.py --model g1d-7.2b
else
    echo "RWKV model found in $MODEL_DIR"
fi

# Optional --config argument (defaults to config.toml)
CONFIG="${1:-config.toml}"

echo ""
echo "Configuration:"
echo "  - Config file: $CONFIG"
echo "  - Model: RWKV7-G1 7.2B (14.4 GB)"
echo "  - GPU 0: RWKV inference"
echo "  - GPU 1: T5 edit model"
echo "  - Log file: $LOGFILE"
echo ""

# Launch with nohup
echo "Launching experiment..."
nohup uv run python main.py --config "$CONFIG" > "$LOGFILE" 2>&1 &

# Get PID
PID=$!
echo $PID > logs/experiment.pid

echo ""
echo "====================================================================="
echo "Experiment launched with PID: $PID"
echo "====================================================================="
echo ""
echo "Commands:"
echo "  Monitor progress:"
echo "    tail -f $LOGFILE"
echo ""
echo "  Check GPU usage:"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "  Stop experiment:"
echo "    kill $PID"
echo ""
echo "====================================================================="
