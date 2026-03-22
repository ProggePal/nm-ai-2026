#!/bin/bash
set -e

echo "Bygger Docker-sandbox for Mac CPU..."
docker build -t nmai-sandbox-mac -f Dockerfile.sandbox.mac .

echo "Klargjør test-data..."
mkdir -p test_data/images
mkdir -p test_output

if [ -d "dataset/images/val" ]; then
    cp dataset/images/val/*.jpg test_data/images/ 2>/dev/null || true
else
    python3 -c "from PIL import Image; Image.new('RGB', (1000, 1000)).save('test_data/images/img_00000.jpg')"
fi

ZIP_FILE="CLEAN_FINAL_SUBMISSION.zip"
if [ ! -f "$ZIP_FILE" ]; then
    ./build_submission.sh
    ZIP_FILE="NM_Submission_NorgesGruppen_100_percent.zip"
fi

rm -rf test_app
mkdir -p test_app
unzip -q "$ZIP_FILE" -d test_app/

if [ ! -f "test_app/run.py" ]; then
    echo "❌ KRITISK FEIL: run.py ligger IKKE på rotnivå i zip-filen!"
    exit 1
fi

rm -f test_output/predictions.json

echo "Kjører Mac CPU sandbox-testen (4 CPUs, 8g Memory, ingen internett)..."
docker run --rm \
    --cpus="4.0" \
    --memory="8g" \
    --network none \
    -v $(pwd)/test_app:/app \
    -v $(pwd)/test_data/images:/data/images:ro \
    -v $(pwd)/test_output:/output \
    -w /app \
    nmai-sandbox-mac \
    python run.py --input /data/images --output /output/predictions.json

if [ -f "test_output/predictions.json" ]; then
    echo "✅ predictions.json ble opprettet. Struktur og minne-test vellykket!"
else
    echo "❌ Feil: predictions.json ble ikke opprettet!"
    exit 1
fi
