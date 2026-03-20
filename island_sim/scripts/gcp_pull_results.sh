#!/bin/bash
# Pull grid search results from GCP VM back to local machine
set -euo pipefail

PROJECT_ID="ai-nm26osl-1886"
VM_NAME="island-sim-search"
ZONE="europe-north1-a"

gcloud config set project "$PROJECT_ID"

echo "Pulling best_params.json..."
gcloud compute scp "$VM_NAME":~/island_sim/data/rounds/best_params.json data/rounds/best_params.json --zone="$ZONE"

echo "Pulling grid_search results..."
mkdir -p data/grid_search
gcloud compute scp --recurse "$VM_NAME":~/island_sim/data/grid_search/ data/grid_search/ --zone="$ZONE"

echo "Done! Best params:"
python -c "
import json
d = json.loads(open('data/rounds/best_params.json').read())
print(f'  Score: {d[\"score\"]}')
print(f'  Source: {d.get(\"source\", \"unknown\")}')
"

echo ""
echo "To delete the VM:"
echo "  gcloud compute instances delete $VM_NAME --zone=$ZONE"
