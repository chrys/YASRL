#!/bin/bash

# Fetch configuration files from production server
set -e

REMOTE_USER="chrys"
REMOTE_HOST="myVPS3"

FILES=(
    "/etc/nginx/sites-enabled/fasolaki.com"
    "/etc/systemd/system/yasrl-api.service"
    "/etc/systemd/system/yasrl-ui.service"
)

echo "Fetching configuration files from $REMOTE_USER@$REMOTE_HOST..."
echo ""

# Create backup directory
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# === SCP Operation Loop ===
for FILE in "${FILES[@]}"; do
    FILENAME=$(basename "$FILE")
    
    # Backup existing file if it exists
    if [ -f "$FILENAME" ]; then
        echo "📦 Backing up existing $FILENAME to $BACKUP_DIR/"
        cp "$FILENAME" "$BACKUP_DIR/$FILENAME"
    fi
    
    # Fetch from remote
    echo "📥 Fetching $FILE from $REMOTE_HOST..."
    scp "$REMOTE_USER@$REMOTE_HOST:$FILE" "$FILENAME"
    
    if [ $? -eq 0 ]; then
        echo "✅ $FILENAME fetched successfully."
    else
        echo "❌ Failed to fetch $FILENAME."
        exit 1
    fi
    echo ""
done

echo "✅ All files fetched successfully!"
echo "Backups stored in: $BACKUP_DIR"
