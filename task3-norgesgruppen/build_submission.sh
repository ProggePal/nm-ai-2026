#!/bin/bash
cd "$(dirname "$0")"

# Bygger endelig zip
rm -rf submission_final
mkdir -p submission_final

cp run.py submission_final/
cp best.onnx submission_final/          # run.py loads best.onnx via onnxruntime
cp feature_bank.json submission_final/
cp resnet50_offline.pth submission_final/

cd submission_final
zip -q -r ../NM_Submission_NorgesGruppen_100_percent.zip ./*
cd ..
echo "100% submission zip er bygget!"
