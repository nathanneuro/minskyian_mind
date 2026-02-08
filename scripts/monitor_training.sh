#!/bin/bash
# Monitor ongoing training on the remote Lambda server

SERVER="nathan@nathan-lambda.taila16957.ts.net"
REMOTE_DIR="/media/external-drive/minsky"

echo "====================================================================="
echo "Minsky Training Monitor (${SERVER})"
echo "====================================================================="

# Check if experiment is running
echo ""
PID=$(ssh $SERVER "cat $REMOTE_DIR/logs/experiment.pid 2>/dev/null")
if [ -n "$PID" ]; then
    RUNNING=$(ssh $SERVER "ps -p $PID -o pid= 2>/dev/null")
    if [ -n "$RUNNING" ]; then
        echo "Training is RUNNING (PID: $PID)"
    else
        echo "Training process not found (PID $PID exited)"
    fi
else
    echo "No PID file found"
fi

# Latest log
echo ""
echo "====================================================================="
echo "Latest Log"
echo "====================================================================="
LOGFILE=$(ssh $SERVER "ls -t $REMOTE_DIR/outputs/logs/*.log 2>/dev/null | head -1")
if [ -n "$LOGFILE" ]; then
    echo "  File: $(basename "$LOGFILE")"
    echo ""
    ssh $SERVER "tail -30 '$LOGFILE'"
else
    echo "  No log files yet"
fi

# Latest training data
echo ""
echo "====================================================================="
echo "Training Data"
echo "====================================================================="
TRAINFILE=$(ssh $SERVER "ls -t $REMOTE_DIR/data/train_data/batch_*.jsonl 2>/dev/null | head -1")
if [ -n "$TRAINFILE" ]; then
    PAIR_COUNT=$(ssh $SERVER "wc -l < '$TRAINFILE'")
    echo "  File: $(basename "$TRAINFILE")  ($PAIR_COUNT pairs)"
    echo "  Last 5 entries:"
    ssh $SERVER "tail -5 '$TRAINFILE'" | while read -r line; do
        echo "    $(echo "$line" | cut -c1-120)..."
    done
else
    echo "  No training data files yet"
fi

# GPU status
echo ""
echo "====================================================================="
echo "GPU Status"
echo "====================================================================="
ssh $SERVER "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" 2>/dev/null | \
    awk -F', ' '{printf "  GPU %s: %s | Util: %s | Mem: %s / %s | Temp: %s\n", $1, $2, $3, $4, $5, $6}'

# Commands
echo ""
echo "====================================================================="
echo "Commands"
echo "====================================================================="
echo "  SSH in:           ssh $SERVER"
echo "  Follow log:       ssh $SERVER \"tail -f $LOGFILE\""
if [ -n "$PID" ]; then
    echo "  Kill training:    ssh $SERVER \"kill $PID\""
fi
echo "====================================================================="
