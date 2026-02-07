#!/bin/bash
# Launch training run with nohup

echo "====================================================================="
echo "Launching Training Run"
echo "====================================================================="

# Create logs directory
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/train_${TIMESTAMP}.log"

echo ""
echo "Configuration:"
echo "  - Training script: train.py"
echo "  - Log file: $LOGFILE"
echo ""

# Launch with nohup - UPDATE THE COMMAND BELOW
nohup uv run python train.py > "$LOGFILE" 2>&1 &

# Get PID
PID=$!
echo "Training launched with PID: $PID"
echo $PID > logs/training.pid

echo ""
echo "====================================================================="
echo "Commands:"
echo "====================================================================="
echo "  Monitor progress:"
echo "    tail -f $LOGFILE"
echo ""
echo "  Check GPU usage:"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "  Stop training:"
echo "    kill $PID"
echo ""
echo "  View in wandb:"
echo "    https://wandb.ai/YOUR_ENTITY/YOUR_PROJECT"
echo ""
echo "====================================================================="
