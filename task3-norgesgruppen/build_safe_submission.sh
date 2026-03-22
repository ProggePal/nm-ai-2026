#!/bin/bash
cd /Users/blekkulf/workspace/norgesgruppen-data/

echo "Bygger fail-safe submission zip..."
rm -rf submission_safe
mkdir -p submission_safe

# Bruker run_safe.py og doper den om til run.py
cp run_safe.py submission_safe/run.py
cp best.pt submission_safe/

cd submission_safe
zip -q -r ../NM_Submission_NorgesGruppen_SAFE.zip ./*
cd ..
echo "NM_Submission_NorgesGruppen_SAFE.zip bygget!"
