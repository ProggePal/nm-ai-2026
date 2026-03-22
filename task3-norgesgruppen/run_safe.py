try:
    __import__("warnings").filterwarnings("ignore")
    print("Starter script (SAFE MODE)")
except Exception:
    pass

def run():
    try:
        j = __import__("json")
        Path = __import__("pathlib").Path
        Image = __import__("PIL.Image").Image
        
        import sys
        
        args = sys.argv
        input_dir = "/data/images"
        output_file = "/output/predictions.json"
        
        if "--input" in args:
            input_dir = args[args.index("--input") + 1]
        if "--output" in args:
            output_file = args[args.index("--output") + 1]
            
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        YOLO = __import__("ultralytics").YOLO
        model = YOLO("best.pt")
        
        predictions = []
        
        for img_path in Path(input_dir).glob("*.jpg"):
            print(f"Behandler: {img_path.name}")
            
            # Trekk ut image_id, f.eks. img_00130.jpg -> 130
            img_id_str = "".join(filter(str.isdigit, img_path.stem))
            image_id = int(img_id_str) if img_id_str else 0
            
            img = Image.open(str(img_path))
            results = model.predict(source=img, conf=0.25, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = x2 - x1
                    h = y2 - y1
                    conf = float(box.conf[0])
                    
                    predictions.append({
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [x1, y1, w, h],
                        "score": conf
                    })
                    
        with open(output_file, "w") as f:
            j.dump(predictions, f, indent=4)
            
        print("Vellykket prosessering, lagret fil:", output_file)
        
    except Exception as e:
        print("KATASTROFAL FEIL, SKRIVER TOM JSON. ERROR:", str(e))
        with open(output_file, "w") as f:
            j.dump([], f)

if __name__ == "__main__":
    run()
