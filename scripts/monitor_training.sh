#!/bin/bash
# Monitor ongoing training

# Get latest log file
LOGFILE=$(ls -t logs/train_*.log 2>/dev/null | head -1)
PIDFILE="logs/training.pid"

if [ -z "$LOGFILE" ]; then
    echo "No training log found"
    exit 1
fi

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Training is RUNNING (PID: $PID)"
    else
        echo "Training process not found (PID: $PID was saved but process died)"
    fi
else
    echo "No PID file found"
fi

echo ""
echo "====================================================================="
echo "Latest Progress"
echo "====================================================================="
tail -30 "$LOGFILE"

echo ""
echo "====================================================================="
echo "GPU Status"
echo "====================================================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
        awk -F', ' '{printf "GPU %s: %s\n  Utilization: %s%%, Memory: %s%% (%s / %s MB), Temp: %s°C\n", $1, $2, $3, $4, $5, $6, $7}'
else
    echo "nvidia-smi not found (no GPU or not NVIDIA)"
fi

echo ""
echo "====================================================================="
echo "Commands"
echo "====================================================================="
echo "  Follow log:       tail -f $LOGFILE"
if [ -f "$PIDFILE" ]; then
    echo "  Kill training:    kill $(cat $PIDFILE)"
fi
echo "====================================================================="
