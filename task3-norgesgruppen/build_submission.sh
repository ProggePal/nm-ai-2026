#!/bin/bash
cd /Users/blekkulf/workspace/norgesgruppen-data/

# Bygger endelig zip
rm -rf submission_final
mkdir -p submission_final

cp run.py submission_final/
cp best.pt submission_final/
cp feature_bank.json submission_final/
cp resnet50_offline.pth submission_final/

cd submission_final
zip -q -r ../NM_Submission_NorgesGruppen_100_percent.zip ./*
cd ..
echo "100% submission zip er bygget!"
