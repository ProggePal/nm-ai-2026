#!/bin/bash
# Pull results from all cluster VMs, merge, and keep global best.
# Reads VM locations from data/cluster_config.json (written by gcp_cluster.sh).
#
# Usage:
#     bash scripts/gcp_cluster_pull.sh
set -euo pipefail

PROJECT_ID="ai-nm26osl-1886"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../data/cluster_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "No cluster config found at $CONFIG_FILE"
    echo "Run gcp_cluster.sh first, or create the config manually."
    exit 1
fi

gcloud config set project "$PROJECT_ID"
mkdir -p data/grid_search

NUM_VMS=$(python3 -c "import json; print(len(json.load(open('$CONFIG_FILE'))))")

# --- Check which VMs are done ---
echo "=== Checking cluster status ==="
ALL_DONE=true
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    VM_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))[$idx]['name'])")
    VM_ZONE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))[$idx]['zone'])")
    RUNNING=$(gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" -- "ps aux | grep 'grid_search_rust' | grep -v grep | wc -l" 2>/dev/null || echo "?")
    if [ "$RUNNING" = "0" ]; then
        echo "  $VM_NAME ($VM_ZONE): DONE"
    elif [ "$RUNNING" = "?" ]; then
        echo "  $VM_NAME ($VM_ZONE): UNREACHABLE"
        ALL_DONE=false
    else
        echo "  $VM_NAME ($VM_ZONE): STILL RUNNING"
        ALL_DONE=false
    fi
done

if [ "$ALL_DONE" = "false" ]; then
    echo ""
    echo "Not all VMs are done. Pull anyway? (y/N)"
    read -r answer
    if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
        echo "Aborting."
        exit 0
    fi
fi

# --- Pull results from each VM ---
echo ""
echo "=== Pulling results ==="
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    VM_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))[$idx]['name'])")
    VM_ZONE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))[$idx]['zone'])")
    echo "  Pulling from $VM_NAME ($VM_ZONE)..."

    # Pull grid search results
    gcloud compute scp --recurse "$VM_NAME":~/island_sim/data/grid_search/ data/grid_search/ --zone="$VM_ZONE" 2>/dev/null || echo "    No grid_search results from $VM_NAME"

    # Pull best_params as a VM-specific file
    gcloud compute scp "$VM_NAME":~/island_sim/data/rounds/best_params.json "data/grid_search/best_params_vm${idx}.json" --zone="$VM_ZONE" 2>/dev/null || echo "    No best_params from $VM_NAME"
done

# --- Merge: find global best ---
echo ""
echo "=== Merging results ==="
python3 -c "
import json
import glob

best_score = -float('inf')
best_file = None
best_data = None

for path in sorted(glob.glob('data/grid_search/best_params_vm*.json')):
    try:
        data = json.loads(open(path).read())
        score = data.get('score', -999)
        print(f'  {path}: score={score:.1f}')
        if score > best_score:
            best_score = score
            best_file = path
            best_data = data
    except Exception as e:
        print(f'  {path}: ERROR - {e}')

if best_data:
    best_data['source'] = 'cluster_grid_search'
    best_data['merged_from'] = best_file
    with open('data/rounds/best_params.json', 'w') as f:
        json.dump(best_data, f, indent=2)
    print(f'')
    print(f'  Global best: {best_score:.1f} (from {best_file})')
    print(f'  Saved to data/rounds/best_params.json')
else:
    print('  No valid results found!')
"

echo ""
echo "=== Done ==="
echo ""
echo "To delete the cluster:"
for idx in $(seq 0 $(( NUM_VMS - 1 ))); do
    VM_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))[$idx]['name'])")
    VM_ZONE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))[$idx]['zone'])")
    echo "  gcloud compute instances delete $VM_NAME --zone=$VM_ZONE -q"
done
