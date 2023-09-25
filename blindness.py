import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load a pre-trained MobileNetV2 model (you can choose a different model depending on your needs)
base_model = MobileNetV2(weights='imagenet', include_top=False)

# Add custom layers for binary classification (e.g., blindness vs. non-blindness)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Load your trained model weights (if you have them)
# model.load_weights('your_model_weights.h5')

# Load and preprocess an image (replace 'your_image.jpg' with your image file)
img_path = '/content/blindness eye.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
prediction = model.predict(x)

# Define a threshold for classifying blindness (adjust as needed)
threshold = 0.5

if prediction[0][0] >= threshold:
    print("Blindness detected.")
else:
    print("No blindness detected.")
