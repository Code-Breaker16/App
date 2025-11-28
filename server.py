import os
import json
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
# Allow CORS for local testing
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"  # Your custom trained model

# Initialize Model
print(f"🔄 Loading Local Model: {MODEL_PATH}...")
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("✅ Model Loaded Successfully!")
    else:
        # Fallback to standard YOLO if custom one isn't there yet
        print(f"⚠️ '{MODEL_PATH}' not found. Downloading standard YOLOv8n for testing...")
        model = YOLO("yolov8n.pt")
        print("✅ Standard YOLOv8n Loaded (It might detect people/cars instead of trees until you train it!)")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "ReLeaf Local AI Server is Running! 🌲"

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if not model:
        return jsonify({"error": "Model not active."}), 500

    try:
        # 1. Receive Image
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"}), 400

        # 2. Decode Base64
        img_str = data['image']
        if "base64," in img_str:
            img_str = img_str.split("base64,")[1]
        
        img_bytes = base64.b64decode(img_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 3. Run YOLO Inference
        # conf=0.15 low confidence threshold to catch more trees
        results = model.predict(img, conf=0.15) 

        # 4. Format Results for Frontend
        detected_objects = []
        result = results[0]
        
        img_h, img_w = img.shape[:2]

        for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            
            # Normalize to 0-1000 scale
            # Frontend expects: [ymin, xmin, ymax, xmax]
            norm_box = [
                int((y1 / img_h) * 1000),
                int((x1 / img_w) * 1000),
                int((y2 / img_h) * 1000),
                int((x2 / img_w) * 1000)
            ]

            # Health Logic (Simulated based on confidence)
            # In a real custom model, you could train classes like "healthy_tree", "dead_tree"
            health = "healthy"
            if score < 0.6: health = "stressed"
            if score < 0.3: health = "dead"

            detected_objects.append({
                "box": norm_box,
                "health": health
            })

        return jsonify({
            "trees": detected_objects,
            "lowGreenCoverDescription": f"Local AI Analysis: Detected {len(detected_objects)} objects."
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Server starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
