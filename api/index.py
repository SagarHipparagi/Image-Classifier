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

model = None

@app.route('/api/favicon.png')
def serve_favicon():
    return send_from_directory(app.root_path, 'favicon.png')

def get_model():
    """Lazy load the model"""
    global model
    if model is None:
        print("Loading MobileNetV2 model...")
        model = MobileNetV2(weights="imagenet")
        print("Model loaded successfully!")
    return model

def preprocess_image(image):
    """Preprocess image for model prediction"""
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] > 3:
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
        return None

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "API is running (Lazy Loading Enabled)"}), 200

@app.route("/", methods=["GET"])
def home():
    try:
        html_file = os.path.join(os.path.dirname(__file__), 'static.html')
        if os.path.exists(html_file):
            with open(html_file, 'r', encoding='utf-8') as f:
                return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
        else:
            return "static.html not found", 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/classify", methods=["GET", "POST", "OPTIONS"])
def classify():
    """Classify an uploaded image"""
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Lazy load model here
        current_model = get_model()
        
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400
            
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        predictions = classify_image(current_model, image)
        
        if predictions is None:
            return jsonify({"error": "Failed to classify image"}), 500
        
        results = []
        for _, label, score in predictions:
            results.append({
                "label": label,
                "confidence": f"{score:.2%}"
            })
        
        return jsonify({"status": "success", "predictions": results}), 200
    
    except Exception as e:
        print(f"ERROR in classify: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
