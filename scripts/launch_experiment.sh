#!/bin/bash
# Launch Minsky Society of Mind experiment
# Server: 2x 4090 GPUs, 500GB RAM

set -e

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

echo ""
echo "Configuration:"
echo "  - Model: RWKV7-G1 7.2B (14.4 GB)"
echo "  - GPU 0: RWKV inference"
echo "  - GPU 1: T5 edit model (if --use-t5)"
echo "  - Log file: $LOGFILE"
echo ""

# Default prompt
PROMPT="${1:-What is the most promising approach to measuring consciousness in AI systems?}"

echo "Prompt: $PROMPT"
echo ""

# Launch with nohup
echo "Launching experiment..."
nohup uv run python main.py \
    --max-steps 10 \
    --summarizer-interval 5 \
    --prompt "$PROMPT" \
    > "$LOGFILE" 2>&1 &

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
