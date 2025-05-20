import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# Load model using a safe relative path
model_path = os.path.join(os.path.dirname(__file__), 'pneumonia_detection_model.h5')
if not os.path.isfile(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)
model = tf.keras.models.load_model(model_path)

# Set test image path (adjusted to relative path)
test_image_path = os.path.join(os.path.dirname(__file__), 'test', 'akash.jpeg')

def predict_image(model, img_path, img_size=(150,150)):
    # Check if image file exists
    if not os.path.isfile(img_path):
        print(f"Error: Image file not found at {img_path}")
        sys.exit(1)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image file {img_path}")
        sys.exit(1)
    
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    prob = model.predict(img)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    print(f"Prediction: {label} (prob={prob:.3f})")

predict_image(model, test_image_path)
