#!/bin/bash

# training server has 500 GB of RAM and 2x 4090s. plan accordingly.

# Configuration
SERVER="nathan@nathan-lambda.taila16957.ts.net"
REMOTE_DIR="/media/external-drive/PROJECTNAME"
LOCAL_DIR="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Deployment Script ===${NC}"

# Create remote directory
echo -e "${YELLOW}Creating remote directory...${NC}"
ssh $SERVER "mkdir -p $REMOTE_DIR"

# Copy project files
echo -e "${YELLOW}Copying project files to server...${NC}"
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='.venv/' \
    --exclude='wandb/' \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='brain_map/' \
    --exclude='logs/' \
    $LOCAL_DIR/ $SERVER:$REMOTE_DIR/






