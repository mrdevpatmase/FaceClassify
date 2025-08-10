import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

EMBEDDINGS_FILE = "embeddings.npy"
LABELS_FILE = "labels.npy"
MODEL_FILE = "svm_face_classifier.joblib"
ENCODER_FILE = "label_encoder.joblib"
SCALER_FILE = "scaler.joblib"

X = np.load(EMBEDDINGS_FILE)
y = np.load(LABELS_FILE)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

base_svm = SVC(kernel="linear", probability=True, random_state=42)
clf = CalibratedClassifierCV(base_svm, cv=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

joblib.dump(clf, MODEL_FILE)
joblib.dump(encoder, ENCODER_FILE)
joblib.dump(scaler, SCALER_FILE)
print("\n💾 Model, encoder, and scaler saved.")
print("\n✅ SVM model trained and saved.")