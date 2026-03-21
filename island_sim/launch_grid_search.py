import json
import subprocess
import sys

with open('data/cluster_config.json') as f:
    config = json.load(f)

procs = []
for vm in config:
    name = vm['name']
    zone = vm['zone']
    seed = vm['seed']
    
    cmd = f'''curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env && ln -sf ~/island_sim/.venv ~/env && export VIRTUAL_ENV=~/env && export PATH="$VIRTUAL_ENV/bin:$PATH" && cd ~/island_sim/rust && maturin develop --release && cd ~/island_sim && nohup python -u scripts/grid_search_rust.py --candidates 50000 --mc-runs 50 --top 20 --seed {seed} > search.log 2>&1 &'''
    
    full_cmd = ['gcloud', 'compute', 'ssh', name, '--zone', zone, '--command', cmd]
    print(f'Starting {name} in {zone} with seed {seed}...')
    p = subprocess.Popen(full_cmd)
    procs.append((name, p))

failed = False
for name, p in procs:
    p.wait()
    if p.returncode != 0:
        print(f'{name} failed with exit code {p.returncode}')
        failed = True
    else:
        print(f'{name} started successfully.')

if failed:
    sys.exit(1)
