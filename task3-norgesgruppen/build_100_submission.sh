#!/bin/bash
cd /Users/blekkulf/workspace/norgesgruppen-data/

echo "Bygger fail-safe 100% submission zip..."
rm -rf submission_100
mkdir -p submission_100

cp run_100_submission.py submission_100/run.py
cp best.pt submission_100/
cp feature_bank.json submission_100/
cp resnet50_offline.pth submission_100/

cd submission_100
zip -q -r ../NM_Submission_NorgesGruppen_100_SAFE.zip ./*
cd ..
echo "NM_Submission_NorgesGruppen_100_SAFE.zip bygget!"
