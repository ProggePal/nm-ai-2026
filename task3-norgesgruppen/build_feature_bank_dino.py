import os
import json
import numpy as np
from pathlib import Path
import cv2
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

def main():
    print("Setter opp DINOv2 for Supercharged Metric Learning...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Kjorer pa: {device}")
    
    # Laster inn Meta sin DINOv2-modell fra PyTorch Hub
    # Vi bruker ViT-Large for maksimal detaljkunnskap om produktet (opplost features)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = model.to(device)
    model.eval()
    
    # DINOv2 forventer at bildene er normalisert pa ImageNet-maner, og opplosningen 
    # ma vare delelig med 14 (patch size). Vi velger 224x224 (16x16 patches).
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    products_dir = Path("/Users/blekkulf/workspace/norgesgruppen-data/products")
        
    print("Bygger DINOv2-database for alle 356 produkter...")
    
    with open("/Users/blekkulf/workspace/norgesgruppen-data/dataset/train/annotations.json", "r") as f:
        coco = json.load(f)
        
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    feature_bank = {}
    product_folders = [f for f in products_dir.iterdir() if f.is_dir()]
    
    for folder in tqdm(product_folders):
        product_name_or_ean = folder.name
        
        category_id = -1
        for cat_id, cat_name in categories.items():
            if product_name_or_ean in cat_name or cat_name in product_name_or_ean:
                category_id = cat_id
                break
                
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        
        if not images:
            continue
            
        folder_embeddings = []
        
        with torch.no_grad():
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                tensor = transform(img).unsqueeze(0).to(device)
                
                # DINOv2 ekstraherer features
                embedding = model(tensor).squeeze(0).cpu().numpy()
                
                # L2-normalisering
                embedding = embedding / np.linalg.norm(embedding)
                folder_embeddings.append(embedding)
                
        if folder_embeddings:
            # Siden DINOv2 forstar produktets "konsept", fungerer gjennomsnittsvektoren ekstremt bra!
            mean_embedding = np.mean(folder_embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            
            feature_bank[product_name_or_ean] = {
                "category_id": category_id,
                "embedding": mean_embedding.tolist()
            }
            
    out_path = Path("/Users/blekkulf/workspace/norgesgruppen-data/feature_bank_dinov2.json")
    with open(out_path, "w") as f:
        json.dump(feature_bank, f)
        
    print(f"\nFerdig! Bygget DINOv2-basert feature bank for {len(feature_bank)} produkter.")
    print(f"Lagret til: {out_path}")

if __name__ == "__main__":
    main()
