#!/bin/bash

# BR_AI_N Voice Assistant - Backup and Push Script
# Creates a timestamped zip backup and pushes to git

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/isaac/codes/indu_brain_v2_1"
BACKUP_DIR="/home/isaac/codes/backups"
PROJECT_NAME="indu_brain_v2_1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/${PROJECT_NAME}_${TIMESTAMP}.zip"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  BR_AI_N Backup & Push Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Step 1: Create backup directory if it doesn't exist
echo -e "${YELLOW}[1/4] Creating backup directory...${NC}"
mkdir -p "$BACKUP_DIR"

# Step 2: Create zip backup (excluding unnecessary files)
echo -e "${YELLOW}[2/4] Creating zip backup...${NC}"
zip -r "$BACKUP_FILE" . \
    -x "*.pyc" \
    -x "*__pycache__*" \
    -x "*.git/*" \
    -x "*indu_vectorstore/*" \
    -x "*.env" \
    -x "*node_modules/*" \
    -x "*.venv/*" \
    -x "*venv/*" \
    -x "*.zip" \
    -x "*.log"

echo -e "${GREEN}   Backup created: ${BACKUP_FILE}${NC}"

# Get backup size
BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo -e "${GREEN}   Backup size: ${BACKUP_SIZE}${NC}"

# Step 3: Git operations
echo -e "${YELLOW}[3/4] Git status...${NC}"
git status --short

echo ""
echo -e "${YELLOW}[4/4] Git push...${NC}"

# Check if there are changes to commit
if [[ -n $(git status --porcelain) ]]; then
    echo -e "${YELLOW}   Uncommitted changes detected. Add and commit first?${NC}"
    read -p "   Enter commit message (or press Enter to skip commit): " COMMIT_MSG

    if [[ -n "$COMMIT_MSG" ]]; then
        git add .
        git commit -m "$COMMIT_MSG"
        echo -e "${GREEN}   Changes committed.${NC}"
    fi
fi

# Push to remote
git push
echo -e "${GREEN}   Pushed to remote.${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Backup & Push Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  Backup: ${BACKUP_FILE}"
echo -e "  Size:   ${BACKUP_SIZE}"
echo ""
