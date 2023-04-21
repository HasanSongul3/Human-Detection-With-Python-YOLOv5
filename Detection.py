import time
import torch
import cv2
from PIL import Image

path = "File path/file name"

# Fotoğrafı açın ve bir görüntü nesnesi oluşturun
foto = Image.open(path)

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc.
# model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/best.pt')  # custom trained model

# Inference
results = model(foto)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

results.xyxy[0]  # im predictions (tensor)
results.pandas().xyxy[0]  # im predictions (pandas)
