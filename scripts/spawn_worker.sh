#!/bin/bash
# Spawn a worker for a GitHub issue with proper monitoring setup
# Usage: ./spawn_worker.sh <issue_number> <branch_name> <worktree_dir>

set -e

ISSUE_NUMBER=$1
BRANCH_NAME=$2
WORKTREE_DIR=$3

if [ -z "$ISSUE_NUMBER" ] || [ -z "$BRANCH_NAME" ] || [ -z "$WORKTREE_DIR" ]; then
    echo "Usage: $0 <issue_number> <branch_name> <worktree_dir>"
    exit 1
fi

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKERS_DIR="$PROJECT_ROOT/.workers"

mkdir -p "$WORKERS_DIR"

# Get issue title
ISSUE_TITLE=$(gh issue view "$ISSUE_NUMBER" --json title -q '.title' 2>/dev/null || echo "Issue #$ISSUE_NUMBER")

# Generate worker ID
WORKER_ID=$(date +%s | md5sum | head -c 8)

echo "Spawning worker $WORKER_ID for Issue #$ISSUE_NUMBER: $ISSUE_TITLE"
echo "Branch: $BRANCH_NAME"
echo "Worktree: $WORKTREE_DIR"

# Create worktree
git worktree add "$WORKTREE_DIR" -b "$BRANCH_NAME" || {
    echo "Failed to create worktree, trying to use existing branch..."
    git worktree add "$WORKTREE_DIR" "$BRANCH_NAME"
}

# Initialize worker log
LOG_FILE="$WORKERS_DIR/${WORKER_ID}.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker $WORKER_ID started" > "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Issue #$ISSUE_NUMBER: $ISSUE_TITLE" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Branch: $BRANCH_NAME" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worktree: $WORKTREE_DIR" >> "$LOG_FILE"

# Update status file
STATUS_FILE="$WORKERS_DIR/status.json"
if [ ! -f "$STATUS_FILE" ]; then
    echo '{"workers": {}, "history": []}' > "$STATUS_FILE"
fi

# Use Python to update JSON properly
python3 << EOF
import json
from datetime import datetime

status_file = "$STATUS_FILE"
with open(status_file) as f:
    status = json.load(f)

now = datetime.now().isoformat()
status['workers']['$WORKER_ID'] = {
    'worker_id': '$WORKER_ID',
    'issue_number': $ISSUE_NUMBER,
    'issue_title': '''$ISSUE_TITLE''',
    'branch_name': '$BRANCH_NAME',
    'worktree_path': '$WORKTREE_DIR',
    'status': 'running',
    'started_at': now,
    'updated_at': now,
    'current_phase': 'initializing',
    'progress_notes': []
}

with open(status_file, 'w') as f:
    json.dump(status, f, indent=2)

print(f"Worker {status['workers']['$WORKER_ID']['worker_id']} registered")
EOF

echo ""
echo "Worker initialized. Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  python scripts/worker_monitor.py status"
echo ""
echo "Worker ID: $WORKER_ID"
