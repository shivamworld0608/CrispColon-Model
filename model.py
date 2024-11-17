from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf  # Replace with torch if using PyTorch
import io

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
# Replace 'path/to/your_model' with the actual path to your trained model file.
model = tf.keras.models.load_model('C:/Users/DELL/OneDrive/Desktop/clg shit/SE Lab/ColonCancerDetection/trained.keras')

# Define a preprocessing function to resize and normalize the image as required by the model
def preprocess_image(image, target_size=(224, 224)):  # Replace (224, 224) with your model's input shape
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize if your model expects normalized input
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

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
    app.run(debug=False)
