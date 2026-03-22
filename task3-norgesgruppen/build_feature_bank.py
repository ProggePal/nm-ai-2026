import os
import json
import numpy as np
from pathlib import Path
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

def main():
    print("Setter opp ResNet50 for Metric Learning (Product Classification)...")
    # Bruker GPU hvis tilgjengelig, ellers CPU (siden dette kjores lokalt na)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Kjorer pa: {device}")
    
    # Laster inn ResNet50 pre-trent pa ImageNet for a bruke som Feature Extractor
    # Fjerner det siste klassifikasjonslaget slik at vi far en 2048-dimensjonal embedding-vektor i stedet for 1000 klasser
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity() 
    model = model.to(device)
    model.eval()
    
    # Standard transformasjoner for ImageNet-modeller
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    products_dir = Path("/Users/blekkulf/workspace/norgesgruppen-data/products")
    if not products_dir.exists():
        print("Finner ikke produktbildene. Pass pa at de er pakket ut i ~/workspace/norgesgruppen-data/products")
        return
        
    print("Bygger referanse-database for alle 356 produkter...")
    
    # Laster COCO-annoteringene for a fa map'en mellom 'id' (0-355) og 'navn/EAN'
    with open("/Users/blekkulf/workspace/norgesgruppen-data/dataset/train/annotations.json", "r") as f:
        coco = json.load(f)
        
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    # Siden produktmappene heter feks '8720608009268' (EAN) eller produktnavn, 
    # ma vi matche mappenavnene mot klassene fra COCO.
    # Dette datasettet kan ha litt messy mapping, sa vi sjekker alle mapper.
    
    feature_bank = {}
    
    product_folders = [f for f in products_dir.iterdir() if f.is_dir()]
    print(f"Fant {len(product_folders)} produktmapper.")
    
    for folder in tqdm(product_folders):
        product_name_or_ean = folder.name
        
        # Finn tilsvarende kategori-ID fra COCO (0-355)
        # Siden vi ikke vet eksakt navne-konvensjon fra utpakkingen enna, lagrer vi det med mappenavn og
        # kan gjore en mapping under inferens.
        category_id = -1
        for cat_id, cat_name in categories.items():
            if product_name_or_ean in cat_name or cat_name in product_name_or_ean:
                category_id = cat_id
                break
                
        if category_id == -1:
            # Hvis vi ikke finner en eksakt match, la oss anta mappenavn KAN vare ID eller at vi kan fuzzy-matche senere
            # For na lagrer vi med folder name som nokkel.
            pass
            
        # Hent ut hovedbildet (eller alle bilder for a lage en robust gjennomsnittsvektor per produkt)
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        
        if not images:
            continue
            
        folder_embeddings = []
        
        with torch.no_grad():
            for img_path in images:
                # Les bilde med OpenCV (bgr), konverter til RGB
                img = cv2.imread(str(img_path))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Transform og legg til batch-dimensjon
                tensor = transform(img).unsqueeze(0).to(device)
                
                # Generer 2048-dimensjonal vektor
                embedding = model(tensor).squeeze(0).cpu().numpy()
                
                # Normaliser embeddingen (L2) slik at Cosine Similarity bare er et dot-produkt!
                embedding = embedding / np.linalg.norm(embedding)
                folder_embeddings.append(embedding)
                
        if folder_embeddings:
            # Gjennomsnittsvektor for produktet basert pa alle vinklene (main, front, back, etc)
            mean_embedding = np.mean(folder_embeddings, axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            
            # Lagre i banken
            feature_bank[product_name_or_ean] = {
                "category_id": category_id,
                "embedding": mean_embedding.tolist()
            }
            
    # Lagre databasen til en JSON fil (dette blir "hjernen" vi tar med i submission-zip'en)
    out_path = Path("/Users/blekkulf/workspace/norgesgruppen-data/feature_bank.json")
    with open(out_path, "w") as f:
        json.dump(feature_bank, f)
        
    print(f"\nFerdig! Bygget feature bank for {len(feature_bank)} produkter.")
    print(f"Lagret til: {out_path}")

if __name__ == "__main__":
    main()
