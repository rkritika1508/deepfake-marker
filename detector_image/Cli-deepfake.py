import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'  # Avoids protobuf binary version mismatch warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: stops some extra logs

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define image size used in training
IMG_SIZE = (256, 256)
MODEL_PATH = "/Users/kritikarupauliha/Downloads/deepfake-detector-cnn-model.keras"
CLASS_NAMES = ["fake", "real"] 

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    """Load and preprocess the image."""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize like in training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array, img

def predict(image_path):
    """Predict class for the given image."""
    
    img_array, original_img = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return CLASS_NAMES[predicted_class], confidence, original_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image using a trained Keras model.")
    parser.add_argument("image", help="Path to the image file")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File '{args.image}' not found.")
        exit(1)

    label, confidence, original_img = predict(args.image)
    print(f"Prediction: {label} (confidence: {confidence:.2f})")

    # Display image with prediction
    plt.imshow(original_img)
    plt.title(f"Prediction: {label} ({confidence:.2f})", color="green" if confidence > 0.5 else "red")
    plt.axis("off")
    plt.show()
