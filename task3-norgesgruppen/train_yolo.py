from ultralytics import YOLO

# Last ned en forhåndstrent YOLO11 nano-modell for hastighet (vi oppgraderer senere)
model = YOLO('yolo11n.pt')

# Tren modellen
results = model.train(
    data='/Users/blekkulf/workspace/norgesgruppen-data/yolo_train/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu' # Bruker CPU lokalt inntil du har GPU klar
)
