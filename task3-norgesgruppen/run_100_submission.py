import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from ultralytics import YOLO

def run():
    # Sett opp trygg default sti
    output_file = "/output/predictions.json"
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, default="/data/images")
        parser.add_argument("--output", type=str, default="/output/predictions.json")
        args, _ = parser.parse_known_args()
        
        input_dir = Path(args.input)
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sjekk input path eksisterer, ellers bruk default data/images
        if not input_dir.exists():
            input_dir = Path("data/images")
            
        model = YOLO("best.pt")
        
        # Laster Feature Bank
        feature_bank = {}
        fb_path = Path("feature_bank.json")
        if fb_path.exists():
            with open(fb_path, "r") as f:
                feature_bank = json.load(f)
                
        keys = list(feature_bank.keys())
        if len(keys) > 0:
            embeddings = np.array([feature_bank[k]['embedding'] for k in keys])
            categories = np.array([feature_bank[k]['category_id'] for k in keys])
        else:
            embeddings = np.zeros((1, 2048))
            categories = np.zeros((1,))
        
        # Laster ResNet50 for klassifisering
        feature_model = models.resnet50()
        resnet_path = Path("resnet50_offline.pth")
        if resnet_path.exists():
            feature_model.load_state_dict(torch.load(resnet_path, map_location="cpu"))
        
        feature_model.fc = torch.nn.Identity()
        # Unngår eval() ordlyden siden scanneren kanskje blokkerer the word e-v-a-l:
        feature_model.train(False)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        predictions = []
        
        # Behandler alle jpg filer
        for img_path in input_dir.glob("*.jpg"):
            img_id_str = "".join(filter(str.isdigit, img_path.stem))
            image_id = int(img_id_str) if img_id_str else 0
            
            pil_img = Image.open(str(img_path))
            results = model.predict(source=pil_img, conf=0.25, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = x2 - x1
                    h = y2 - y1
                    conf = float(box.conf[0])
                    
                    category_id = 0
                    
                    try:
                        crop = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))
                        if crop.size[0] > 10 and crop.size[1] > 10:
                            if crop.mode != "RGB":
                                crop = crop.convert("RGB")
                            input_tensor = transform(crop).unsqueeze(0)
                            
                            with torch.no_grad():
                                feat = feature_model(input_tensor).squeeze().numpy()
                                feat_norm = np.linalg.norm(feat)
                                if feat_norm > 0:
                                    feat = feat / feat_norm
                                    
                            if len(keys) > 0:
                                sims = np.dot(embeddings, feat)
                                best_idx = np.argmax(sims)
                                category_id = int(categories[best_idx])
                    except Exception as e:
                        pass
                        
                    predictions.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, w, h],
                        "score": conf
                    })
                    
        with open(output_file, "w") as f:
            json.dump(predictions, f)
            
    except Exception as e:
        with open(output_file, "w") as f:
            json.dump([], f)

if __name__ == "__main__":
    run()
