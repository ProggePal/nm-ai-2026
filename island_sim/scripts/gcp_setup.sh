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
VM_NAME="island-sim-search"
ZONE="europe-north1-a"         # Low latency from Norway
MACHINE_TYPE="n2-highcpu-96"   # 96 vCPUs, 96 GB RAM
PROJECT_DIR="island_sim"

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
# Create a tarball of just what we need (no .env, no .git)
TMPTAR=$(mktemp /tmp/island_sim_XXXX.tar.gz)
tar czf "$TMPTAR" \
    --exclude='.git' \
    --exclude='.env' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/grid_search' \
    -C "$(dirname "$(cd "$(dirname "$0")/.." && pwd)")" \
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
echo "2. Run the grid search:"
echo "   cd ~/island_sim && ~/env/bin/python scripts/grid_search.py --workers 90 --candidates 2000 --mc-runs 50"
echo ""
echo "3. When done, pull results back to your laptop:"
echo "   gcloud compute scp $VM_NAME:~/island_sim/data/rounds/best_params.json data/rounds/best_params.json --zone=$ZONE"
echo "   gcloud compute scp --recurse $VM_NAME:~/island_sim/data/grid_search data/ --zone=$ZONE"
echo ""
echo "4. Delete the VM when finished:"
echo "   gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
