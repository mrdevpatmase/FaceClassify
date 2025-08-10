import os
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

DATASET_DIR = r"dataset"
PROCESSED_DIR = r"processed_faces"
MAPPING_FILE = "image_paths.txt"

os.makedirs(PROCESSED_DIR, exist_ok=True)

mtcnn = MTCNN(image_size=160, margin=20)
processed_paths = []

for root, _, files in os.walk(DATASET_DIR):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(root, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                face = mtcnn(img)
                if face is not None:
                    save_path = os.path.join(PROCESSED_DIR, filename)
                    transforms.ToPILImage()(face).save(save_path)
                    label = os.path.basename(root)
                    processed_paths.append((save_path, label))
                    print(f"✅ Saved {save_path}")
                else:
                    print(f"❌ No face found in {img_path}")
            except Exception as e:
                print(f"⚠ Error processing {img_path}: {e}")

with open(MAPPING_FILE, "w", encoding="utf-8") as f:
    for path, label in processed_paths:
        f.write(f"{path}|{label}\n")

print(f"✅ Saved {len(processed_paths)} images to {PROCESSED_DIR}")