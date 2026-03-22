import requests
import json
import os
import urllib.request
from pathlib import Path
import time
import sys

API_KEY = os.environ.get("KASSALAPP_API_KEY", "")

def search_and_download(ean_or_name, category_id, save_dir):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }
    
    # Prøver å fjerne evt vekt for å få bredere søk hvis det kræsjer "Zalo Oppvask 500ml" -> "Zalo Oppvask"
    clean_name = ean_or_name.split(" ")[0:2]
    clean_name = " ".join(clean_name)
    url = f"https://kassal.app/api/v1/products?search={clean_name}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"[{category_id}] Feilet API-kall for {clean_name}: {response.status_code}")
            return False
            
        data = response.json()
        products = data.get("data", [])
        
        if not products:
            print(f"[{category_id}] Fant ikke {clean_name} i Kassal.app")
            return False
            
        best_match = products[0]
        image_url = best_match.get("image")
        
        if not image_url:
            print(f"[{category_id}] Fant produkt, men intet bilde for {clean_name}")
            return False
            
        cat_dir = Path(save_dir) / str(category_id)
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        img_name = f"kassal_{best_match['id']}.jpg"
        save_path = cat_dir / img_name
        
        if save_path.exists():
            return True
            
        urllib.request.urlretrieve(image_url, str(save_path))
        print(f"[{category_id}] ✅ Lastet ned: {clean_name}")
        return True
        
    except Exception as e:
        print(f"[{category_id}] Exception for {clean_name}: {e}")
        return False

def main():
    print("Starte Kassal.app Data Augmentation...")
    sys.stdout.flush()
    
    with open("/Users/blekkulf/workspace/norgesgruppen-data/dataset/train/annotations.json", "r") as f:
        coco = json.load(f)
        
    save_dir = "/Users/blekkulf/workspace/norgesgruppen-data/products_augmented"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    # Tar de 20 første som en test først
    for cat in coco["categories"][:50]:
        cat_id = cat["id"]
        name = cat["name"]
        
        if search_and_download(name, cat_id, save_dir):
            success_count += 1
            
        time.sleep(1.2) # Unngå spam (Kassal.app er en liten startup)
        sys.stdout.flush()
        
    print(f"\nFerdig med batch! Beriket med {success_count} nye produktbilder fra Kassal.app!")

if __name__ == "__main__":
    main()
