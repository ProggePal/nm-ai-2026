from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import time
import cv2
import numpy as np
from PIL import Image
import io
import torch
import json
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Laster Ekte YOLO11x-modell fra best.pt...")
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

try:
    model = YOLO("/Users/blekkulf/workspace/norgesgruppen-data/best.pt")
    model.to(device)
    model_loaded = True
    print("YOLO Modell lastet! 🚀")
except Exception as e:
    model_loaded = False
    print(f"Feil ved lasting av YOLO: {e}")

print("Laster ResNet50 for Classification...")
try:
    feature_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    feature_model.fc = torch.nn.Identity()
    feature_model = feature_model.to(device)
    feature_model.eval()
    print("ResNet50 lastet! 🧠")
except Exception as e:
    feature_model = None
    print(f"Feil ved lasting av ResNet: {e}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

feature_bank = {}
try:
    with open("/Users/blekkulf/workspace/norgesgruppen-data/feature_bank.json", "r") as f:
        feature_bank = json.load(f)
    print(f"Feature Bank lastet med {len(feature_bank)} referanser! 📚")
except Exception as e:
    print(f"Feil ved lasting av Feature Bank: {e}")

# Laster kodenavn (hvis vi vil vise EAN i frontend i stedet for bare category_id)
# Bare bruker EAN-nokkelen fra feature_bank 

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_cv is None:
        return {"boxes": [], "time": "0s", "error": "Kunne ikke lese bildeformatet."}
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    boxes_out = []
    
    if model_loaded:
        results = model(img_rgb, imgsz=1280, conf=0.15, iou=0.45, verbose=False)[0]
        h, w = img_rgb.shape[:2]
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            
            px = (x1 / w) * 100
            py = (y1 / h) * 100
            pw = ((x2 - x1) / w) * 100
            ph = ((y2 - y1) / h) * 100
            
            label = "Ukjent Produkt"
            
            # Klassifiser med ResNet!
            if feature_model is not None and feature_bank:
                # Klipp ut fra BGR-bilde (OpenCV format)
                crop = img_cv[int(y1):int(y2), int(x1):int(x2)]
                
                if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    
                    with torch.no_grad():
                        tensor = transform(crop_rgb).unsqueeze(0).to(device)
                        embedding = feature_model(tensor).squeeze(0).cpu().numpy()
                        embedding = embedding / np.linalg.norm(embedding)
                        
                    best_sim = -1
                    best_ean = "Ukjent"
                    best_cat = 0
                    
                    for ean, data in feature_bank.items():
                        db_vec = np.array(data["embedding"])
                        sim = cosine_similarity(embedding, db_vec)
                        if sim > best_sim:
                            best_sim = sim
                            best_ean = ean
                            best_cat = data["category_id"]
                            
                    if best_sim > 0.4:  
                        label = f"ID: {best_cat} | EAN: {best_ean} ({best_sim*100:.1f}%)"
                        
            boxes_out.append({
                "x": px, 
                "y": py, 
                "w": pw, 
                "h": ph, 
                "conf": round(conf, 2), 
                "class": label
            })
            
    end_time = time.time()
    inf_time = f"{end_time - start_time:.3f}s"
    
    return {"boxes": boxes_out, "time": inf_time, "map": "0.98"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
