#!/bin/bash
set -euo pipefail
SEED="$1"
CANDIDATES="$2"
MC_RUNS="$3"
TOP="$4"

pkill -f grid_search_rust 2>/dev/null || true

# Ensure build dependencies are installed (cmaes crate needs fontconfig)
sudo apt-get update -qq && sudo apt-get install -y -qq pkg-config libfontconfig1-dev > /dev/null 2>&1 || true

cd ~
rm -rf island_sim
tar xzf island_sim.tar.gz
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd island_sim/rust
maturin develop --release 2>&1 | tail -1
python -c "import island_sim_core; print('Rust OK')"
cd ~/island_sim
nohup python -u scripts/grid_search_rust.py \
    --candidates "$CANDIDATES" --mc-runs "$MC_RUNS" --top "$TOP" --seed "$SEED" \
    > search.log 2>&1 &
echo "PID: $!"
