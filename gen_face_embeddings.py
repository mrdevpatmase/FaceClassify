import os
from PIL import Image
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

processed_folder = r"C:\Users\devpa\OneDrive\Desktop\FaceClassify\processed_faces"

model = InceptionResnetV1(pretrained='vggface2').eval()

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

embeddings = []
labels = []

for image in os.listdir(processed_folder):
    image_path = os.path.join(processed_folder, image)
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            embedding = model(img_tensor).squeeze().numpy()

        embeddings.append(embedding)

        label = image.split("_")[0].lower()  
        labels.append(label)

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")

np.save("embeddings.npy", np.array(embeddings))
np.save("labels.npy", np.array(labels))

print("✅ Embeddings and labels saved.")
