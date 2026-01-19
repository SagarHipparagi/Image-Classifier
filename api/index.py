import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

@app.route('/api/favicon.png')
def serve_favicon():
    return send_from_directory(app.root_path, 'favicon.png')


# Load model once when the app starts
try:
    print("Loading MobileNetV2 model...")
    model = MobileNetV2(weights="imagenet")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image to 224x224
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    
    # Ensure 3 channels (RGB)
    if img.ndim == 2:  # Grayscale
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] > 3:  # More than 3 channels
        img = img[:, :, :3]
    
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def classify_image(model, image):
    """Classify image using the model"""
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        print(f"Classification error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok", "message": "API is running"}), 200


@app.route("/", methods=["GET"])
def home():
    """Serve the frontend UI from static.html"""
    try:
        # Try to read the static.html file from the same directory
        html_file = os.path.join(os.path.dirname(__file__), 'static.html')
        if os.path.exists(html_file):
            with open(html_file, 'r', encoding='utf-8') as f:
                return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
        else:
            # Fallback to basic HTML if static.html doesn't exist
            return """<!DOCTYPE html><html><head><title>Classifier</title></head><body><h1>Welcome to Image Classifier</h1><p>static.html not found</p></body></html>"""
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/classify", methods=["GET", "POST", "OPTIONS"])
def classify():
    """Classify an uploaded image"""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400
        
        # Read and process the image
        image_data = file.read()
        if not image_data:
            return jsonify({"error": "Empty image file"}), 400
            
        image = Image.open(io.BytesIO(image_data))
        
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Classify the image
        predictions = classify_image(model, image)
        
        if predictions is None:
            return jsonify({"error": "Failed to classify image"}), 500
        
        # Format predictions
        results = []
        for _, label, score in predictions:
            results.append({
                "label": label,
                "confidence": f"{score:.2%}"
            })
        
        response_data = {
            "status": "success",
            "predictions": results
        }
        return jsonify(response_data), 200
    
    except Exception as e:
        print(f"ERROR in classify: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            return jsonify({"error": "Classification failed"}), 500
        except:
            return '{"error": "Server error"}', 500


if __name__ == "__main__":
    app.run(debug=True)
