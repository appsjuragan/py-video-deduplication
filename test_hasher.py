import sys
import os
import numpy as np
import torch
import subprocess
import io
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

FFMPEG = r'C:\msys64\mingw64\bin\ffmpeg.exe'
f1 = r'C:\Users\Administrator\Desktop\sa\5_6291612929515068372.mp4.mp4'
f2 = r'C:\Users\Administrator\Desktop\sa\VID_20231222_043145_290.mp4.mp4'

def get_keyframes(path, num=32):
    cmd = [
        FFMPEG, 
        '-y',
        '-i', path, 
        '-vf', 'select=eq(pict_type\,I),scale=224:224:force_original_aspect_ratio=increase,crop=224:224', 
        '-vsync', 'vfr', 
        '-vframes', str(num), 
        '-f', 'image2pipe', 
        '-vcodec', 'png', 
        '-an',
        'pipe:1'
    ]
    res = subprocess.run(cmd, capture_output=True, creationflags=0x08000000)
    data = res.stdout
    frames = []
    # Simple PNG split
    raw_frames = data.split(b'\x89PNG\r\n\x1a\n')
    for chunk in raw_frames[1:]:
        frames.append(Image.open(io.BytesIO(b'\x89PNG\r\n\x1a\n' + chunk)).convert("RGB"))
    return frames

print(f"Extracting keyframes...")
fr1 = get_keyframes(f1)
fr2 = get_keyframes(f2)
print(f"Extracted: {len(fr1)} / {len(fr2)}")

if not fr1 or not fr2:
    print("Failed to extract keyframes")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier = nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_fp(frames):
    tensors = torch.stack([transform(f) for f in frames]).to(device)
    with torch.no_grad():
        feats = model(tensors).cpu().numpy()
    # Mask pooling
    fp = np.max(feats, axis=0)
    fp = fp - np.mean(fp)
    norm = np.linalg.norm(fp)
    if norm > 0: fp /= norm
    return fp

sim = np.dot(get_fp(fr1), get_fp(fr2))
print(f"--- RESULTS ---")
print(f"SIMILARITY: {sim*100.0:.2f}%")
