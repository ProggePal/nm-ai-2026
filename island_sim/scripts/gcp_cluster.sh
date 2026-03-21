#!/bin/bash
# Spin up a cluster of c4d-highcpu-384 VMs for parallel grid search.
# Each VM explores a different region of parameter space via --seed.
# VMs are spread across GCP regions (quota is 1 c4d per region).
#
# Usage:
#     bash scripts/gcp_cluster.sh                       # 5 VMs, 50k candidates each
#     bash scripts/gcp_cluster.sh --vms 3               # 3 VMs
#     bash scripts/gcp_cluster.sh --candidates 100000   # 100k per VM
set -euo pipefail

# --- Configuration ---
PROJECT_ID="ai-nm26osl-1886"
MACHINE_TYPE="c4d-highcpu-384"
NUM_VMS=5
CANDIDATES=50000
MC_RUNS=50
TOP=20

# Zones spread across regions (1 c4d-384 per region due to quota)
ALL_ZONES=(
    "us-central1-a"
    "europe-west1-b"
    "asia-southeast1-a"
    "us-east1-b"
    "us-east4-a"
    "us-west1-a"
    "asia-northeast1-b"
)

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --vms) NUM_VMS="$2"; shift 2 ;;
        --candidates) CANDIDATES="$2"; shift 2 ;;
        --mc-runs) MC_RUNS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$NUM_VMS" -gt "${#ALL_ZONES[@]}" ]; then
    echo "Error: max ${#ALL_ZONES[@]} VMs (one per region)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Cluster Grid Search ==="
echo "  VMs:         $NUM_VMS × $MACHINE_TYPE"
echo "  Candidates:  $CANDIDATES per VM ($(( NUM_VMS * CANDIDATES )) total)"
echo "  MC runs:     $MC_RUNS"
echo "  Zones:"
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    echo "    VM $idx: ${ALL_ZONES[$idx]}"
done
echo ""

gcloud config set project "$PROJECT_ID"

# --- Step 1: Create all VMs ---
echo "=== Creating $NUM_VMS VMs ==="
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    VM_NAME="island-search-${idx}"
    VM_ZONE="${ALL_ZONES[$idx]}"
    echo "  Creating $VM_NAME in $VM_ZONE..."
    gcloud compute instances create "$VM_NAME" \
        --zone="$VM_ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=debian-12 \
        --image-project=debian-cloud \
        --boot-disk-size=20GB \
        --boot-disk-type=hyperdisk-balanced \
        2>&1 | grep -E 'Created|ERROR|RUNNING' || echo "  $VM_NAME: may already exist, continuing..."
done

echo "Waiting 30s for VMs to boot..."
sleep 30

# --- Step 2: Wait for SSH on all VMs ---
echo "=== Waiting for SSH ==="
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    VM_NAME="island-search-${idx}"
    VM_ZONE="${ALL_ZONES[$idx]}"
    for attempt in $(seq 1 12); do
        if gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" -- "echo ready" 2>/dev/null; then
            echo "  $VM_NAME ($VM_ZONE): SSH ready"
            break
        fi
        if [ "$attempt" -eq 12 ]; then
            echo "  $VM_NAME ($VM_ZONE): SSH failed after 12 attempts"
            exit 1
        fi
        sleep 10
    done
done

# --- Step 3: Package code ---
echo "=== Packaging code ==="
TMPTAR=$(mktemp /tmp/island_sim_XXXX.tar.gz)
tar czf "$TMPTAR" \
    --exclude='.git' \
    --exclude='.env' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/grid_search' \
    --exclude='data/visualizations' \
    --exclude='rust/target' \
    -C "$REPO_ROOT" \
    island_sim

# --- Step 4: Setup each VM (parallel) ---
echo "=== Setting up VMs ==="
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    VM_NAME="island-search-${idx}"
    VM_ZONE="${ALL_ZONES[$idx]}"
    (
        echo "  [$VM_NAME] Uploading code..."
        gcloud compute scp "$TMPTAR" "$VM_NAME":~/island_sim.tar.gz --zone="$VM_ZONE"

        echo "  [$VM_NAME] Installing dependencies..."
        gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" -- bash -s <<'REMOTE_SETUP'
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv python3-dev curl build-essential
python3 -m venv ~/env
~/env/bin/pip install -q numpy cma maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
cd ~
tar xzf island_sim.tar.gz
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd ~/island_sim/rust
maturin develop --release 2>&1
python -c "import island_sim_core; print('Rust backend OK')"
REMOTE_SETUP
        echo "  [$VM_NAME] Ready!"
    ) &
done
wait
rm "$TMPTAR"

# --- Step 5: Launch grid search on each VM ---
echo ""
echo "=== Launching grid search ==="
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    VM_NAME="island-search-${idx}"
    VM_ZONE="${ALL_ZONES[$idx]}"
    SEED="$idx"
    echo "  [$VM_NAME] Starting search with --seed $SEED..."
    gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" -- bash -c "
        cd ~/island_sim && \
        nohup ~/env/bin/python -u scripts/grid_search_rust.py \
            --candidates $CANDIDATES \
            --mc-runs $MC_RUNS \
            --top $TOP \
            --seed $SEED \
            > search.log 2>&1 &
        echo 'PID: '\$!
    "
done

# --- Save cluster config for pull script ---
CONFIG_FILE="$SCRIPT_DIR/../data/cluster_config.json"
mkdir -p "$(dirname "$CONFIG_FILE")"
{
    echo "["
    for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
        COMMA=""
        [ "$idx" -lt "$(( NUM_VMS - 1 ))" ] && COMMA=","
        echo "  {\"name\": \"island-search-${idx}\", \"zone\": \"${ALL_ZONES[$idx]}\", \"seed\": $idx}${COMMA}"
    done
    echo "]"
} > "$CONFIG_FILE"
echo "Cluster config saved to $CONFIG_FILE"

echo ""
echo "=========================================="
echo "  Cluster launched: $NUM_VMS × $MACHINE_TYPE"
echo "  Total candidates: $(( NUM_VMS * CANDIDATES ))"
echo "  Total vCPUs: $(( NUM_VMS * 384 ))"
echo "=========================================="
echo ""
echo "Monitor progress:"
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    echo "  gcloud compute ssh island-search-$idx --zone=${ALL_ZONES[$idx]} -- 'tail -5 ~/island_sim/search.log'"
done
echo ""
echo "Pull results:"
echo "  bash scripts/gcp_cluster_pull.sh"
echo ""
echo "Delete cluster:"
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    echo "  gcloud compute instances delete island-search-$idx --zone=${ALL_ZONES[$idx]} -q"
done
