import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
from class_labels import CLASS_NAMES
import io

# --- Configuration ---
app = Flask(__name__)
# The model file you renamed in Step 1
MODEL_PATH = 'hair_disease_model.keras' 
IMG_WIDTH, IMG_HEIGHT = 224, 224 # Match your training input size

# --- Load Model ---
# Load the model only once when the server starts
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Preprocessing Function ---
def preprocess_image(image_file):
    """Loads, resizes, and normalizes the image for the model."""
    # Read the image file content
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Resize the image
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    # Your model expects normalization to 0-1, 
    # check your training—if it was /255, keep this:
    img_array = img_array / 255.0 
    
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in the request.")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No selected file.")
            
        if file:
            try:
                # 1. Preprocess the image
                processed_image = preprocess_image(file)
                
                if model is None:
                    return render_template('index.html', error="Model is not loaded.")
                
                # 2. Make prediction
                predictions = model.predict(processed_image)
                
                # 3. Get the predicted class index and probability
                predicted_class_index = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                # 4. Map index to class name
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                
                result = {
                    'class_name': predicted_class_name,
                    'confidence': f"{confidence:.2f}%"
                }
                
                return render_template('index.html', result=result)

            except Exception as e:
                return render_template('index.html', error=f"An error occurred during prediction: {e}")

    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the model is loaded before running
    if model:
        app.run(debug=True) # Run with debugging enabled
    else:
        print("Application stopped because the model failed to load.")