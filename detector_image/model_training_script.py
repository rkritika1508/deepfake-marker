import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

train_dir = '/kaggle/input/deepfake-and-real-images/Dataset/Train'
test_dir = '/kaggle/input/deepfake-and-real-images/Dataset/Test'
validation_dir = '/kaggle/input/deepfake-and-real-images/Dataset/Validation'

def normalize_image(image, labels):
    image = tf.cast(image, tf.float32) / 255.0
    return image, labels


IMG_SIZE = (256, 256)
EPOCHS = 20
train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                 label_mode = 'int',
                                                                 batch_size = 32,
                                                                 image_size= IMG_SIZE)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(validation_dir,
                                                                      label_mode = 'int',
                                                                      batch_size = 32,
                                                                      image_size= IMG_SIZE)

test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode = 'int',
                                                                batch_size = 32,
                                                                image_size= IMG_SIZE,
                                                                shuffle = False)

train_data = train_data.map(normalize_image)
validation_data = validation_data.map(normalize_image)
test_data = test_data.map(normalize_image)

model = Sequential([
    Conv2D(filters = 8, kernel_size = 5,input_shape = (256, 256, 3), activation= 'relu'),
    MaxPooling2D(),
    BatchNormalization(),

    Conv2D(filters= 16, kernel_size=4, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Dropout(0.2),

    Conv2D(filters= 32, kernel_size=3, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),

    Conv2D(filters= 64, kernel_size=2, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),

    Conv2D(filters= 128, kernel_size=1, activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Dropout(0.2),

    Flatten(),
    Dropout(0.3),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 1, activation = 'sigmoid')
])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'BinaryCrossentropy',
              metrics=['accuracy', 'precision', 'recall', 'auc'])

hist = model.fit(train_data,
                 epochs = EPOCHS,
                 validation_data = validation_data,
                 validation_steps = int(0.5 * len(validation_data))
                 )

model.evaluate(test_data)
model.save("deepfake-detector-cnn-model.keras")

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal',label='loss')
plt.plot(hist.history['val_loss'], color='orange',label='val_loss')
fig.suptitle('Loss',fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal',label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange',label='val_accuracy')
fig.suptitle('Accuracy',fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['precision'], color='teal', label='precision')
plt.plot(hist.history['val_precision'], color='orange', label='val_precision')
fig.suptitle('Precision', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['recall'], color='teal', label='recall')
plt.plot(hist.history['val_recall'], color='orange', label='val_recall')
fig.suptitle('Recall', fontsize=20)
plt.legend(loc="upper left")
plt.show()

true_labels = []
predictions = []

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
true_labels = []
predictions = []

for images, labels in test_data:
    true_labels.extend(labels.numpy().astype(int)) 
    batch_predictions = (model.predict(images) > 0.5).astype(int).flatten()
    predictions.extend(batch_predictions)

true_labels = np.array(true_labels)
predictions = np.array(predictions)

cm = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
