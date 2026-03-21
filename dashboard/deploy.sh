#!/bin/bash
# Deploy dashboard to Cloud Run (unauthenticated).
# Usage: bash dashboard/deploy.sh
set -euo pipefail

PROJECT_ID="ai-nm26osl-1886"
SERVICE_NAME="astar-dashboard"
REGION="europe-west1"

# Read JWT from .env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JWT_TOKEN=$(grep JWT_TOKEN "$SCRIPT_DIR/../island_sim/.env" | cut -d= -f2-)

echo "=== Deploying $SERVICE_NAME to Cloud Run ==="
echo "  Project: $PROJECT_ID"
echo "  Region:  $REGION"

gcloud config set project "$PROJECT_ID"

# Build and deploy in one step
gcloud run deploy "$SERVICE_NAME" \
    --source "$SCRIPT_DIR" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars "JWT_TOKEN=$JWT_TOKEN,DATA_DIR=/app/data" \
    --memory 256Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 2 \
    --timeout 60 \
    --port 8080

echo ""
echo "=== Deployed! ==="
URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format 'value(status.url)')
echo "Dashboard URL: $URL"
