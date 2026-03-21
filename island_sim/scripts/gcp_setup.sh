#!/bin/bash
# GCP setup script for Astar Island grid search (Rust backend)
set -euo pipefail

# --- Configuration ---
PROJECT_ID="ai-nm26osl-1886"
VM_NAME="island-sim-search"
ZONE="europe-north1-a"
MACHINE_TYPE="n2d-highcpu-224"  # 224 vCPUs, 224 GB RAM

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
echo "Waiting for SSH to become available..."
for i in $(seq 1 12); do
    if gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- "echo ready" 2>/dev/null; then
        break
    fi
    echo "  Attempt $i/12 — retrying in 10s..."
    sleep 10
done

echo ""
echo "=== Step 3: Install Python, Rust, and dependencies ==="
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- bash -s <<'REMOTE_INSTALL'
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv python3-dev curl build-essential
python3 -m venv ~/env
~/env/bin/pip install -q numpy cma maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
echo "Rust and Python installed"
REMOTE_INSTALL

echo ""
echo "=== Step 4: Upload project code and data ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TMPTAR=$(mktemp /tmp/island_sim_XXXX.tar.gz)
tar czf "$TMPTAR" \
    --exclude='.git' \
    --exclude='.env' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/grid_search' \
    --exclude='rust/target' \
    -C "$REPO_ROOT" \
    island_sim

gcloud compute scp "$TMPTAR" "$VM_NAME":~/island_sim.tar.gz --zone="$ZONE"
rm "$TMPTAR"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- bash -s <<'REMOTE_EXTRACT'
cd ~
tar xzf island_sim.tar.gz
ls -la island_sim/
REMOTE_EXTRACT

echo ""
echo "=== Step 5: Build Rust extension ==="
gcloud compute ssh "$VM_NAME" --zone="$ZONE" -- bash -s <<'REMOTE_BUILD'
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd ~/island_sim/rust
maturin develop --release 2>&1
echo "Rust extension built!"
python -c "import island_sim_core; print('Rust backend loaded OK')"
REMOTE_BUILD

echo ""
echo "=========================================="
echo "VM is ready with Rust backend!"
echo "=========================================="
echo ""
echo "1. SSH into the VM:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "2. Run the grid search (nohup so it survives SSH disconnect):"
echo "   cd ~/island_sim && nohup ~/env/bin/python -u scripts/grid_search_rust.py --candidates 100000 --mc-runs 50 --top 20 > search.log 2>&1 &"
echo ""
echo "3. Check progress:"
echo "   tail -f ~/island_sim/search.log"
echo ""
echo "4. Pull results (from laptop):"
echo "   bash scripts/gcp_pull_results.sh"
echo ""
echo "5. Delete VM:"
echo "   gcloud compute instances delete $VM_NAME --zone=$ZONE"
