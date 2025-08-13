import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # Avoids protobuf binary version mismatch warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: stops some extra logs

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ------------------------------
# 1. Load Model
# ------------------------------
MODEL_PATH = "/Users/kritikarupauliha/Downloads/deepfake-detector-cnn-model (1).keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------
# 2. Read CSV (format: 'filename','label')
# ------------------------------
csv_path = "/Users/kritikarupauliha/Downloads/true-media-deepfakes-in-the-wild/image-metadata-publish.csv"
img_dir = "/Users/kritikarupauliha/Downloads/true-media-deepfakes-in-the-wild/image-data/"      # folder where images are stored
df = pd.read_csv(csv_path)

# Ensure labels are binary (0/1)
df['label'] = df['Ground Truth'].map({'Fake': 0, 'Real': 1})  # adjust if your CSV uses different names
df = df[df['Filename'] != '#NAME?']

# ------------------------------
# 3. Load & Preprocess Images
# ------------------------------
X = []
y_true = []

for _, row in df.iterrows():
    img_path = img_dir + row['Filename']
    try:
        img = tf.keras.utils.load_img(img_path, target_size=(256, 256))  # resize
        img_array = tf.keras.utils.img_to_array(img) / 255.0  # normalize
        X.append(img_array)
        y_true.append(row['label'])
    except Exception as e:
        print(f"Skipping {img_path} due to error: {e}")


X = np.array(X)
y_true = np.array(y_true)

# ------------------------------
# 4. Make Predictions
# ------------------------------
y_pred_prob = model.predict(X)
y_pred = (y_pred_prob > 0.5).astype("int32")
print(y_pred)

# ------------------------------
# 5. Calculate Metrics
# ------------------------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC: {auc:.3f}")

# ------------------------------
# 6. Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ------------------------------
# 7. ROC Curve (optional)
# ------------------------------
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
