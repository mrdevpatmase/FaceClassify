import os
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN

input_folder = r"C:\Users\devpa\OneDrive\Desktop\FaceClassify\dataset"
output_folder = r"C:\Users\devpa\OneDrive\Desktop\FaceClassify\processed_faces"

os.makedirs(output_folder, exist_ok=True)

detector = MTCNN()

def extract_and_save_face(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    pixels = np.array(img)

    results = detector.detect_faces(pixels)

    if results:
        for i, result in enumerate(results):
            x, y, width, height = result['box']
            x, y = max(0, x), max(0, y)
            width, height = min(pixels.shape[1] - x, width), min(pixels.shape[0] - y, height)
            face = pixels[y:y + height, x:x + width]

            face_image = Image.fromarray(face)
            output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_face{i + 1}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            face_image.save(output_path)
            print(f"Saved cropped face to {output_path}")
    else:
        print(f"No face detected in {image_path}")



for root, _, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, filename)
            extract_and_save_face(image_path)
            print("Face extraction completed.")
