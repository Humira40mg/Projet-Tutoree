# midas_module.py

import torch
import cv2
import numpy as np
import urllib.request
import os

# MiDaS v2.1 small
MODEL_TYPE = "MiDaS_small"

# Télécharger le modèle s’il n'existe pas déjà
if not os.path.exists("midas_small.pt"):
    print("Téléchargement du modèle MiDaS v2.1 Small...")
    urllib.request.urlretrieve(
        "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
        "midas_small.pt"
    )

# Charger le modèle
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.load_state_dict(torch.load("midas_small.pt", map_location=torch.device('cpu')))
midas.eval()

# Transformation
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

def estimate_depth(frame):
    # Prétraitement
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueewe(0)

    with torch.no_grad():
        prediction = midas(input_tensor)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convertir en numpy
    depth_map = prediction.cpu().numpy()
    return depth_map
