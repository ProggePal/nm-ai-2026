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
NUM_VMS=6
CANDIDATES=50000
MC_RUNS=50
TOP=20

# Zones spread across regions (1 c4d-384 per region due to quota)
# us-west1 has 0 C4D quota — excluded
ALL_ZONES=(
    "us-central1-a"
    "europe-west1-b"
    "asia-southeast1-a"
    "us-east1-b"
    "us-east4-a"
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
        --image-family=island-sim \
        --image-project="$PROJECT_ID" \
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
        if gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --command="echo ready" 2>/dev/null; then
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
# --- Step 3a: Create per-VM launcher script ---
# This avoids complex quoting in gcloud ssh --command
LAUNCH_SCRIPT="$SCRIPT_DIR/_vm_launch.sh"
cat > "$LAUNCH_SCRIPT" << 'LAUNCHER'
#!/bin/bash
set -euo pipefail
SEED="$1"
CANDIDATES="$2"
MC_RUNS="$3"
TOP="$4"

pkill -f grid_search_rust 2>/dev/null || true

# Ensure build dependencies are installed (cmaes crate needs fontconfig)
sudo apt-get update -qq && sudo apt-get install -y -qq pkg-config libfontconfig1-dev > /dev/null 2>&1 || true

cd ~
rm -rf island_sim
tar xzf island_sim.tar.gz
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd island_sim/rust
maturin develop --release 2>&1 | tail -1
python -c "import island_sim_core; print('Rust OK')"
cd ~/island_sim
nohup python -u scripts/grid_search_rust.py \
    --candidates "$CANDIDATES" --mc-runs "$MC_RUNS" --top "$TOP" --seed "$SEED" \
    > search.log 2>&1 &
echo "PID: $!"
LAUNCHER
chmod +x "$LAUNCH_SCRIPT"

echo "=== Packaging code ==="
TMPTAR=$(mktemp /tmp/island_sim_XXXXXXXXXX)
mv "$TMPTAR" "${TMPTAR}.tar.gz"
TMPTAR="${TMPTAR}.tar.gz"
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

# --- Step 4: Upload, build, and launch search (parallel per VM) ---
# Each VM does everything in one SSH session: extract, build, start search.
# No cross-VM dependencies — each VM is fully independent.
# Uses a function with local vars to avoid race conditions in parallel subshells.
setup_vm() {
    local vm_name="$1" vm_zone="$2" seed="$3"
    echo "  [$vm_name] Uploading code + launcher..."
    gcloud compute scp "$TMPTAR" "$vm_name":~/island_sim.tar.gz --zone="$vm_zone"
    gcloud compute scp "$LAUNCH_SCRIPT" "$vm_name":~/vm_launch.sh --zone="$vm_zone"

    echo "  [$vm_name] Building and launching (seed=$seed)..."
    gcloud compute ssh "$vm_name" --zone="$vm_zone" --command="bash ~/vm_launch.sh $seed $CANDIDATES $MC_RUNS $TOP"
    echo "  [$vm_name] Done!"
}

echo "=== Setting up and launching on all VMs ==="
PIDS=()
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    setup_vm "island-search-${idx}" "${ALL_ZONES[$idx]}" "$idx" &
    PIDS+=($!)
done

# Wait for all and check for failures
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done
rm -f "$TMPTAR" "$LAUNCH_SCRIPT"

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED VM(s) failed setup. Check output above."
    exit 1
fi

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
