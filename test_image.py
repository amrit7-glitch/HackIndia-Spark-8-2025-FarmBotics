import tensorflow as tf
import numpy as np
import cv2
import json

# Load the trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Load class names
with open("class_names.json", "r") as f:
    class_indices = json.load(f)

# Reverse the class_indices dict to get index-to-name mapping
class_names = [None] * len(class_indices)
for name, index in class_indices.items():
    class_names[index] = name

# Image path (change this to your test image)
img_path = "D:\\Test_plb.JPG"

# Read and preprocess the image
img = cv2.imread(img_path)
img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_index = np.argmax(prediction)

print("Predicted class index:", predicted_index)
print("Predicted class name:", class_names[predicted_index])
