#this is the main file for this program
import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add scripts directory to path to import preprocess
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from preprocess import preprocess_points_to_28x28
from keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load Model and Labels
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'scripts', 'models', 'airwrite_model.h5')
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'scripts', 'models', 'labels.npy')

model = None
labels = None
gesture_model = None
gesture_labels = None

try:
    model = load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH)
    print(f"[INFO] Air-writing model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load air-writing model: {e}")

try:
    GESTURE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'scripts', 'models', 'gesture_model.h5')
    GESTURE_LABELS_PATH = os.path.join(os.path.dirname(__file__), 'scripts', 'models', 'gesture_labels.npy')
    if os.path.exists(GESTURE_MODEL_PATH):
        gesture_model = load_model(GESTURE_MODEL_PATH)
        gesture_labels = np.load(GESTURE_LABELS_PATH, allow_pickle=True)
        print(f"[INFO] Gesture model loaded successfully from {GESTURE_MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load gesture model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or labels is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.json
    points = data.get('points', [])
    
    if len(points) < 5:
        return jsonify({'error': 'Not enough points', 'prediction': ''}), 400
        
    try:
        # Preprocess points to 28x28 image
        img = preprocess_points_to_28x28(points)
        inp = img[None, ...] # shape (1, 28, 28, 1)
        
        # Predict
        pred = model.predict(inp, verbose=0)[0]
        ci = int(np.argmax(pred))
        conf = float(pred[ci])
        ch = str(labels[ci])
        
        return jsonify({
            'prediction': ch,
            'confidence': conf
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_gesture', methods=['POST'])
def predict_gesture():
    if gesture_model is None or gesture_labels is None:
        return jsonify({'error': 'Gesture model not loaded. Train it first using scripts/train_gestures.py'}), 500
        
    data = request.json
    features = data.get('features', [])
    
    if len(features) != 63: # 21 landmarks * 3 coords
        return jsonify({'error': 'Invalid feature shape', 'prediction': ''}), 400
        
    try:
        inp = np.array(features)[None, ...]
        pred = gesture_model.predict(inp, verbose=0)[0]
        ci = int(np.argmax(pred))
        conf = float(pred[ci])
        word = str(gesture_labels[ci])
        
        return jsonify({
            'prediction': word,
            'confidence': conf
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
