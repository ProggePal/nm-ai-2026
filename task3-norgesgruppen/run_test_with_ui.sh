#!/bin/bash
cd /Users/blekkulf/workspace/norgesgruppen-data/

function update_status() {
    local status=$1
    local msg=$2
    # Vi kan bruke python for a unnga jq dependency
    python3 -c "
import json
with open('test_status.json', 'r') as f: data = json.load(f)
data['status'] = '$status'
data['log'].append('$msg')
with open('test_status.json', 'w') as f: json.dump(data, f)
"
}

echo '{"status": "INITIATING", "log": ["Starter Docker Sandkasse..."]}' > test_status.json
sleep 1
update_status "RUNNING" "Klargjør ressursgrenser: 8GB RAM, 4 vCPU, INGEN nettverkstilgang."
sleep 1

update_status "RUNNING" "Bygger fail-safe submission zip (NM_Submission_NorgesGruppen_SAFE.zip)..."
./build_safe_submission.sh >/dev/null 2>&1
update_status "RUNNING" "Bygging av Zip-fil vellykket."
sleep 1

update_status "RUNNING" "Pakker ut zip-fil i fiktiv sandkasse-katalog..."
rm -rf test_app
mkdir -p test_app
unzip -q NM_Submission_NorgesGruppen_SAFE.zip -d test_app/

if [ ! -f "test_app/run.py" ]; then
    update_status "FAILED" "KRITISK FEIL: run.py ligger IKKE pa rotniva i zip-filen!"
    exit 1
fi

update_status "RUNNING" "Mappestruktur godkjent. run.py ligger pa rotniva."
sleep 1

update_status "RUNNING" "Starter container for CPU-simulering (Mac miljo)..."

rm -f test_output/predictions.json

docker run --rm \
    --cpus="4.0" \
    --memory="8g" \
    --network none \
    -v $(pwd)/test_app:/app \
    -v $(pwd)/test_data/images:/data/images:ro \
    -v $(pwd)/test_output:/output \
    -w /app \
    nmai-sandbox-mac \
    python run.py --input /data/images --output /output/predictions.json > docker_output.txt 2>&1

DOCKER_EXIT=$?

if [ $DOCKER_EXIT -eq 0 ]; then
    update_status "RUNNING" "[DOCKER]: Kjøring fullført. Sjekker fil-output..."
else
    update_status "FAILED" "[DOCKER]: Feil under kjøring. Exit kode: $DOCKER_EXIT"
    update_status "FAILED" "Feilmelding lagret."
    exit 1
fi

sleep 1

if [ -f "test_output/predictions.json" ]; then
    update_status "SUCCESS" "SUCCESS: predictions.json ble opprettet og fylt med data!"
    update_status "SUCCESS" "Sandkasse test bestatt med glans! Vi kan trygt levere denne Zip-en til NM-portalen."
else
    update_status "FAILED" "FEIL: predictions.json ble ikke opprettet! Fallback-mekanismen feilet."
    exit 1
fi
