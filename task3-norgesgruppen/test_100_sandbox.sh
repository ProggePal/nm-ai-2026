#!/bin/bash
cd /Users/blekkulf/workspace/norgesgruppen-data/

echo "Pakker ut 100% zip-fil i fiktiv sandkasse-katalog..."
rm -rf test_app_100
mkdir -p test_app_100
unzip -q NM_Submission_NorgesGruppen_100_SAFE.zip -d test_app_100/

rm -f test_output/predictions_100_test.json

echo "Kjører container..."
docker run --rm \
    --cpus="4.0" \
    --memory="8g" \
    --network none \
    -v $(pwd)/test_app_100:/app \
    -v $(pwd)/test_data/images:/data/images:ro \
    -v $(pwd)/test_output:/output \
    -w /app \
    nmai-sandbox-mac \
    python run.py --input /data/images --output /output/predictions_100_test.json > docker_100_output.txt 2>&1

DOCKER_EXIT=$?
if [ $DOCKER_EXIT -eq 0 ]; then
    echo "100% Test Vellykket!"
else
    echo "100% Test Feilet!"
fi
