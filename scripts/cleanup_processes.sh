#!/bin/bash
# Cleanup script for stranded processes

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Process Cleanup ===${NC}"

# Kill wandb processes
echo -e "${YELLOW}Killing wandb processes...${NC}"
wandb_pids=$(pgrep -f wandb)
if [ -n "$wandb_pids" ]; then
    echo "Found wandb processes: $wandb_pids"
    pkill -9 -f wandb
    echo -e "${GREEN}✓ Killed wandb processes${NC}"
else
    echo "No wandb processes found"
fi

# Kill pytest processes
echo -e "${YELLOW}Killing pytest processes...${NC}"
pytest_pids=$(pgrep -f pytest)
if [ -n "$pytest_pids" ]; then
    echo "Found pytest processes: $pytest_pids"
    pkill -9 -f pytest
    echo -e "${GREEN}✓ Killed pytest processes${NC}"
else
    echo "No pytest processes found"
fi

# Kill python processes from this project - UPDATE PROJECT NAME
echo -e "${YELLOW}Killing project python processes...${NC}"
project_pids=$(pgrep -f "your-project-name")
if [ -n "$project_pids" ]; then
    echo "Found project processes: $project_pids"
    pkill -9 -f "your-project-name"
    echo -e "${GREEN}✓ Killed project processes${NC}"
else
    echo "No project processes found"
fi

# Kill processes from PID files
if [ -d "logs" ]; then
    echo -e "${YELLOW}Killing processes from PID files...${NC}"
    for pidfile in logs/*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if ps -p $pid > /dev/null 2>&1; then
                echo "Killing process $pid from $pidfile"
                kill -9 $pid 2>/dev/null || true
            fi
            rm "$pidfile"
        fi
    done
    echo -e "${GREEN}✓ Cleaned up PID files${NC}"
fi

echo ""
echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo ""
echo "Remaining Python processes:"
ps aux | grep python | grep -v grep
