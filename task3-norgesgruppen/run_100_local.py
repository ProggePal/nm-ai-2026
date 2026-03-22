import json
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

def run():
    print("Starter FULL PIPELINE (100%) lokal test...")
    input_dir = Path("test_data/images")
    output_file = Path("test_output/predictions_100.json")
    
    model = YOLO("best.pt")
    
    with open("feature_bank.json", "r") as f:
        feature_bank = json.load(f)
        
    print("Forbereder vector space...")
    keys = list(feature_bank.keys())
    embeddings = np.array([feature_bank[k]['embedding'] for k in keys])
    categories = np.array([feature_bank[k]['category_id'] for k in keys])
    
    print("Laster ResNet50...")
    feature_model = resnet50()
    try:
        feature_model.load_state_dict(torch.load("resnet50_offline.pth", map_location="cpu"))
        print("Lokal resnet50_offline.pth lastet.")
    except Exception as e:
        print("Klarte ikke laste offline weights:", e)
        return
        
    feature_model.fc = torch.nn.Identity()
    feature_model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    
    for img_path in input_dir.glob("*.jpg"):
        if "00000" in img_path.name: continue
        print(f"Behandler: {img_path.name}")
        
        img_id_str = "".join(filter(str.isdigit, img_path.stem))
        image_id = int(img_id_str) if img_id_str else 0
        
        orig_img = cv2.imread(str(img_path))
        if orig_img is None: continue
        
        pil_img = Image.open(str(img_path))
        results = model.predict(source=pil_img, conf=0.25, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                conf = float(box.conf[0])
                
                category_id = 0
                max_sim = 0.0
                
                crop = orig_img[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(crop_rgb).unsqueeze(0)
                    
                    with torch.no_grad():
                        feat = feature_model(input_tensor).squeeze().numpy()
                        feat_norm = np.linalg.norm(feat)
                        if feat_norm > 0:
                            feat = feat / feat_norm
                            
                    sims = np.dot(embeddings, feat)
                    best_idx = np.argmax(sims)
                    max_sim = float(sims[best_idx])
                    category_id = int(categories[best_idx])
                            
                predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "score": conf,
                    "similarity": max_sim
                })
                
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
        
    print("100% generering fullført!")

if __name__ == "__main__":
    run()
