import torch
from torchvision.models import resnet50
import os

print("Laster ned ResNet50 vekter for offline-bruk...")
# Last ned ved hjelp av standardfunksjonen, men lagre filen direkte
model = resnet50(weights="IMAGENET1K_V2")
torch.save(model.state_dict(), "resnet50_offline.pth")
print("Lagret til resnet50_offline.pth! Inkluder denne i submission zip-en din.")