#!/bin/bash
set -e

echo "=== KLARGJØR TEST-MILJØ LOKALT ==="
mkdir -p test_data/images
mkdir -p test_output

if [ -d "dataset/images/val" ]; then
    cp dataset/images/val/*.jpg test_data/images/ 2>/dev/null || true
elif [ -d "yolo_train/images/val" ]; then
    cp yolo_train/images/val/*.jpg test_data/images/ 2>/dev/null || true
else
    echo "Lager et tomt testbilde da datasett ikke ble funnet på forventet sted."
    python3 -c "import numpy as np, cv2; cv2.imwrite('test_data/images/img_00000.jpg', np.zeros((1000, 1000, 3), dtype=np.uint8))"
fi

echo "=== KJØRER TESTEN ==="
# Sletter gammel output
rm -f test_output/predictions.json

# Kjører koden lokalt som den ville blitt kjørt i skyen
python3 run.py --input ./test_data/images --output ./test_output/predictions.json

echo "=== EVALUERING ==="
if [ -f "test_output/predictions.json" ]; then
    echo "✅ predictions.json ble opprettet. De første linjene er:"
    head -n 20 test_output/predictions.json
    echo "..."
    echo "✅ Testen var Vellykket!"
else
    echo "❌ Feil: predictions.json ble ikke opprettet i output-mappen!"
    exit 1
fi
