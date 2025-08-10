import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

MAPPING_FILE = "image_paths.txt"
EMBEDDINGS_FILE = "embeddings.npy"
LABELS_FILE = "labels.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

embeddings, labels = [], []
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    for line in f:
        img_path, label = line.strip().split("|")
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(img_tensor).cpu().numpy()[0]
        embeddings.append(emb)
        labels.append(label)
        print(f"✅ Processed {img_path}")

np.save(EMBEDDINGS_FILE, embeddings)
np.save(LABELS_FILE, labels)
print("\n💾 Embeddings & labels saved.")
