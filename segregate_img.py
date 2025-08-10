import os
import shutil
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
import joblib
import numpy as np

TEST_IMAGE = r"C:\Users\devpa\OneDrive\Desktop\FaceClassify SVM\testing_dataset\vk.jpg"
MAPPING_FILE = "image_paths.txt"
MODEL_FILE = "svm_face_classifier.joblib"
ENCODER_FILE = "label_encoder.joblib"
SCALER_FILE = "scaler.joblib"
SEARCH_RESULTS_DIR = "search_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

clf = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)
scaler = joblib.load(SCALER_FILE)

# Preprocess test image
img = Image.open(TEST_IMAGE).convert("RGB")
face = mtcnn(img)
if face is None:
    print("❌ No face detected in test image.")
    exit()

img_tensor = transform(transforms.ToPILImage()(face)).unsqueeze(0).to(device)
with torch.no_grad():
    test_emb = model(img_tensor).cpu().numpy()

test_emb_scaled = scaler.transform(test_emb)
probs = clf.predict_proba(test_emb_scaled)[0]
pred_idx = np.argmax(probs)
confidence = probs[pred_idx]
pred_label = encoder.inverse_transform([pred_idx])[0]

print(f"\n🔍 Predicted: {pred_label} (Confidence: {confidence:.2f})")

if confidence < 0.3:
    print("❌ Low confidence. Aborting copy.")
    exit()

os.makedirs(SEARCH_RESULTS_DIR, exist_ok=True)
dest_folder = os.path.join(SEARCH_RESULTS_DIR, pred_label)
os.makedirs(dest_folder, exist_ok=True)

copied_count = 0

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    for line in f:
        path, label = line.strip().split("|")
        if label.lower() == pred_label.lower():
            filename = os.path.basename(path)
            original_path = os.path.join("dataset", label, filename)
            
            if os.path.exists(original_path):
                shutil.copy(original_path, dest_folder)
                copied_count += 1
            else:
                print(f"⚠️ Original file not found: {original_path}")



print(f"✅ {copied_count} images copied to: {dest_folder}")
