import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Load Data
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")

encoder = LabelEncoder()
y = encoder.fit_transform(labels)
num_classes = len(set(y))
y_cat = tf.keras.utils.to_categorical(y, num_classes)

X = embeddings[..., np.newaxis]

# Build Model
model = models.Sequential([
    keras.layers.Input(shape=(512,1)),
    keras.layers.Conv1D(32, 3, activation='relu'),
    keras.layers.MaxPooling1D(),
    keras.layers.Conv1D(64, 3, activation='relu'),
    keras.layers.MaxPooling1D(),
    keras.layers.Conv1D(128, 3, activation='relu'),
    keras.layers.MaxPooling1D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model and capture history
history = model.fit(X, y_cat, epochs=50, batch_size=32, validation_split=0.1)

# Save model and label encoder
model.save("cnn_face_classifier.h5")
joblib.dump(encoder, "label_encoder.joblib")
print("Model and encoder saved successfully.")

# Evaluate model
print("Evaluation results:", model.evaluate(X, y_cat))
model.summary()

# 🔽 Plot Training & Validation Loss/Accuracy
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='red')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
