from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import requests

# Initialize Flask app
app = Flask(__name__)

# Download the model if it doesn't exist
MODEL_PATH = 'trained.keras'
MODEL_URL = 'https://crispcolon-model.onrender.com/trained.keras'  # Your actual public model URL
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully.")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define a function to check if the uploaded file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a preprocessing function to resize and normalize the image
def preprocess_image(image, target_size=(224, 224)):  # Replace (224, 224) with your model's input shape
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize if your model expects normalized input
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Validate the file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'}), 400

    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Predict with the model
        predictions = model.predict(processed_image)
        
        # Interpret prediction - this depends on your model output
        prediction_label = 'Cancerous' if predictions[0][0] > 0.5 else 'Non-cancerous'

        # Return the prediction result as JSON
        return jsonify({'prediction': prediction_label, 'confidence': float(predictions[0][0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    # Use PORT from the environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
