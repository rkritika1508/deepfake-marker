# Summary – Deepfake detection using CNN

## 1. Problem Statement
- **Goal:** To develop and deploy a machine learning model to accurately detect deepfake images, helping mitigate the spread of manipulated visual media. The model leverages convolutional neural networks (CNNs) trained on real and fake images to learn visual patterns, enabling automated and fast classification.
- **Type:** Binary classification
- **Domain:** Computer vision

---

## 2. Dataset
- **Source:** https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
- **Size:** Training - 140002 images. Validation - 39428 images. Testing - 10905 images.
- **Target Variable:** Real / fake (determine whether the given image is real or fake)
- **Train/Test Split:** Provided in the dataset

---

## 3. Model Architecture
- **Type:** CNN
- **Layers:**
  - Input: Shape = (256, 256, 3) → Preprocessing: rescaling pixel values to [0, 1]
  - Hidden Layers: Convolutional layers with ReLU activation, MaxPooling layers, Dropout layers for regularization, Flatten layer before dense layers
  - Output: Dense layer with sigmoid activation, 1 unit (binary output)
- **Optimizer:** Adam
- **Loss Function:** BinaryCrossentropy
- **Metrics Tracked:** Accuracy, Precision, Recall, AUC
- **Label Mode:** Categorical (2 classes — Real, Fake)

---

## 4. Training Setup
- **Epochs:** 20
- **Batch Size:** 32
- **Learning Rate:** 0.001 (default value)

---

## 5. Results & Metrics
| Metric        | Train Value | Validation Value | Test Value |
|---------------|-------------|------------------|------------|
| Accuracy      |0.986        |0.949             |0.869       |
| Precision     |0.983        |0.924             |0.883       |
| Recall        |0.99         |0.979             |0.848       |
| AUC           |0.999        |0.989             |0.940       |
| F1-score      |0.986        |0.950             |0.866       |
---
## 6. Some graphs based on metrics
- **Loss curve**
<img width="639" height="478" alt="Screenshot 2025-08-12 at 6 41 00 AM" src="https://github.com/user-attachments/assets/5dce0f8e-2c58-4d97-89df-c4b7defa4246" />

- **Accuracy curve**
<img width="639" height="475" alt="Screenshot 2025-08-12 at 6 41 14 AM" src="https://github.com/user-attachments/assets/663ba05c-8ca3-4e0c-b1fe-c8c357ea79b6" />

- **Precision curve**
<img width="639" height="461" alt="Screenshot 2025-08-12 at 6 41 31 AM" src="https://github.com/user-attachments/assets/ef156929-9f5a-4eef-bae3-c7698162e06e" />

- **Recall curve**
<img width="639" height="471" alt="Screenshot 2025-08-12 at 6 41 46 AM" src="https://github.com/user-attachments/assets/e6c37c8e-c9b9-4696-bde9-de1b305069ef" />

- **Confusion matrix**
<img width="578" height="441" alt="Screenshot 2025-08-12 at 6 42 06 AM" src="https://github.com/user-attachments/assets/f7c6c013-5f38-4d2b-a440-72520d9a0fb4" />

---

## 7. Observations
- Validation and test dataset performance is lower than training metrics, suggesting some overfitting. 
- Recall > Precision: Across all sets, recall is consistently higher than precision, meaning the model is slightly more inclined to label an image as "fake" when in doubt. This reduces false negatives but may increase false positives.

---

## 8. Future Improvements
- Use regularization methods such as dropout, weight decay, or data augmentation to reduce overfitting and improve generalization to unseen data.
- Systematically explore learning rates, batch sizes, and optimizer parameters to find settings that improve validation and test performance without overfitting.
- Experiment with architectures of varying depth and capacity to determine the optimal model.
- Use k-fold cross-validation instead of a single validation split to obtain more robust estimates of performance and reduce the risk of overfitting to a specific validation set.
- Qualitative analysis of misclassified examples.
---





