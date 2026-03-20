#!/bin/bash
# GCP setup script for Astar Island grid search
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - GCP project set (gcloud config set project YOUR_PROJECT_ID)
#
# Usage:
#   1. Run this script to create the VM and deploy code
#   2. SSH in and run the grid search
#   3. Pull results back when done

set -euo pipefail

# --- Configuration ---
PROJECT_ID="ai-nm26osl-1886"
VM_NAME="island-sim-search"
ZONE="europe-north1-a"         # Low latency from Norway
MACHINE_TYPE="n2-highcpu-96"   # 96 vCPUs, 96 GB RAM
PROJECT_DIR="island_sim"

echo "=== Step 0: Set project ==="
gcloud config set project "$PROJECT_ID"

echo ""
echo "=== Step 1: Create VM ==="
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-ssd

echo ""
echo "=== Step 2: Wait for VM to be ready ==="
sleep 15

echo ""
echo "=== Step 3: Install Python and dependencies ==="
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    sudo apt-get update -qq && \
    sudo apt-get install -y -qq python3 python3-pip python3-venv && \
    python3 -m venv ~/env && \
    ~/env/bin/pip install -q numpy cma
"

echo ""
echo "=== Step 4: Upload project code and data ==="
# Resolve the project root (script is in island_sim/scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TMPTAR=$(mktemp /tmp/island_sim_XXXX.tar.gz)
tar czf "$TMPTAR" \
    --exclude='.git' \
    --exclude='.env' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/grid_search' \
    -C "$REPO_ROOT" \
    "$PROJECT_DIR"

gcloud compute scp "$TMPTAR" "$VM_NAME":~/island_sim.tar.gz --zone="$ZONE"
rm "$TMPTAR"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    cd ~ && \
    tar xzf island_sim.tar.gz && \
    ls -la island_sim/
"

echo ""
echo "=========================================="
echo "VM is ready! Next steps:"
echo "=========================================="
echo ""
echo "1. SSH into the VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "2. Run the grid search (nohup so it survives SSH disconnect):"
echo "   cd ~/island_sim && nohup ~/env/bin/python scripts/grid_search.py --workers 90 --candidates 2000 --mc-runs 50 > search.log 2>&1 &"
echo ""
echo "3. Check progress (can disconnect and reconnect anytime):"
echo "   tail -f ~/island_sim/search.log"
echo ""
echo "4. When done, exit SSH and pull results back to your laptop:"
echo "   bash scripts/gcp_pull_results.sh"
echo ""
echo "5. Delete the VM when finished:"
echo "   gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
