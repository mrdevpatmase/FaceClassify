import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load model and encoder
model = load_model("cnn_face_classifier.h5")
encoder = joblib.load("label_encoder.joblib")

# Load embeddings and pick one
embeddings = np.load("embeddings.npy")  
embedding = embeddings[0]               

# Reshape for prediction
embedding = embedding[np.newaxis, ..., np.newaxis]  # shape: (1, 512, 1)

# Predict
pred = model.predict(embedding)
class_index = np.argmax(pred)
label = encoder.inverse_transform([class_index])

print("Predicted Person:", label[0])
