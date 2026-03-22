import time
import subprocess
import re
from datetime import datetime

html_file = "/Users/blekkulf/workspace/norgesgruppen-data/dashboard.html"

def get_yolo_epoch():
    cmd = "export PATH=/opt/homebrew/share/google-cloud-sdk/bin:$PATH && gcloud compute ssh blekkulf@nmai-a100-dl-trainer --project=ai-nm26osl-1886 --zone=europe-west4-a --command='tail -50 train.log | grep -Eo \"^[ ]*[0-9]+/[0-9]+\" | tail -1'"
    try:
        out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        if "/" in out:
            curr, tot = map(int, out.split('/'))
            return curr, tot
    except:
        pass
    return None, 100

def get_kassal_progress():
    try:
        out = subprocess.check_output("cat /tmp/kassal_augment.log | grep 'Lastet ned' | wc -l", shell=True, text=True).strip()
        return int(out)
    except:
        pass
    return None

while True:
    curr_epoch, tot_epoch = get_yolo_epoch()
    kassal_prog = get_kassal_progress()
    
    with open(html_file, 'r') as f:
        content = f.read()
        
    now = datetime.now().strftime("%d.%m.%Y - %H:%M:%S")
    content = re.sub(r'<div class="status">.*</div>', f'<div class="status"><div class="dot"></div> Sist oppdatert: {now}</div>', content)
        
    if curr_epoch is not None:
        pct = int((curr_epoch / tot_epoch) * 100)
        content = re.sub(r'Epoch [0-9]+ / [0-9]+', f'Epoch {curr_epoch} / {tot_epoch}', content)
        content = re.sub(r'<div class="progress-fill green" style="width: [0-9]+%;"></div>', f'<div class="progress-fill green" style="width: {pct}%;"></div>', content)
        
    if kassal_prog is not None:
        # Erstatt hele blue-bar section med kassal progress
        pct = int((kassal_prog / 356) * 100)
        content = re.sub(r'[0-9]+ / 356 mapper', f'{kassal_prog} / 356 mapper', content)
        # Regex er litt brittle her, bruker bare replace:
        import sys
        
    with open(html_file, 'w') as f:
        f.write(content)
        
    time.sleep(10)
