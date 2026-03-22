#!/bin/bash
set -e

# Oppdatert med ressursbegrensninger for L4-sandbox miljø: 4 vCPU, 8 GB RAM, no network.
echo "Bygger Docker-sandbox miljø..."
docker build -t nmai-sandbox -f Dockerfile.sandbox .

echo "Klargjør test-data..."
mkdir -p test_data/images
mkdir -p test_output

# Sjekker om det finnes bilder
if [ -d "dataset/images/val" ]; then
    cp dataset/images/val/*.jpg test_data/images/ 2>/dev/null || true
elif [ -d "yolo_train/images/val" ]; then
    cp yolo_train/images/val/*.jpg test_data/images/ 2>/dev/null || true
else
    # Lag noen dummy bilder hvis ingen finnes
    echo "Lager tomme testbilder..."
    python3 -c "from PIL import Image; Image.new('RGB', (1000, 1000)).save('test_data/images/img_00000.jpg'); Image.new('RGB', (800, 800)).save('test_data/images/img_00001.jpg')"
fi

# Hvis ZIP filen allerede finnes på Mac-en til Hans, bruker vi den.
ZIP_FILE="CLEAN_FINAL_SUBMISSION.zip"
if [ ! -f "$ZIP_FILE" ]; then
    echo "Finner ikke $ZIP_FILE. Bruker bygge-scriptet til å lage en ny..."
    ./build_submission.sh
    ZIP_FILE="NM_Submission_NorgesGruppen_100_percent.zip"
fi

echo "Pakker ut $ZIP_FILE i app-mappen for sandboxen..."
rm -rf test_app
mkdir -p test_app
unzip -q "$ZIP_FILE" -d test_app/

# Sjekker om run.py faktisk ligger på rotnivå
if [ ! -f "test_app/run.py" ]; then
    echo "❌ KRITISK FEIL: run.py ligger IKKE på rotnivå i zip-filen!"
    echo "Innhold av zip-roten:"
    ls -la test_app/
    exit 1
else
    echo "✅ run.py funnet på rotnivå."
fi

# Fjern eventuell tidligere predictions.json for å sikre en fersk test
rm -f test_output/predictions.json

echo "Kjører sandbox-testen (med sandbox specs: 4 CPUs, 8g Memory, ingen internett)..."
docker run --rm \
    --cpus="4.0" \
    --memory="8g" \
    --gpus all \
    --network none \
    -v $(pwd)/test_app:/app \
    -v $(pwd)/test_data/images:/data/images:ro \
    -v $(pwd)/test_output:/output \
    -w /app \
    nmai-sandbox \
    python3 run.py --input /data/images --output /output/predictions.json

echo "Sjekker resultatet..."
if [ -f "test_output/predictions.json" ]; then
    echo "✅ predictions.json ble opprettet:"
    cat test_output/predictions.json | head -n 20
    echo "..."
    echo "🎉 Test vellykket! Koden fullførte innenfor ressursgrensene."
else
    echo "❌ Feil: predictions.json ble ikke opprettet i output-mappen! Koden krasjet sannsynligvis (f.eks OOM)."
    exit 1
fi
