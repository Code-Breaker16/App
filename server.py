import os
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- CONFIGURATION ---
# Use environment variable for API Key
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "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")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# --- ADVANCED SYSTEM PROMPT FOR ACCURACY ---
SYSTEM_INSTRUCTION = """
You are a highly specialized Forestry AI trained for precision urban tree inventory. 
Your task is to analyze aerial or street-level imagery to detect individual trees and assess their health.

**CRITICAL INSTRUCTIONS:**
1.  **Detection Strategy:**
    -   Do NOT draw one large box around a forest or group of trees.
    -   Identify DISTINCT, INDIVIDUAL tree crowns.
    -   Ignore bushes, grass, or low-lying vegetation. Only detect trees.
    -   If trees overlap, try to estimate the center mass of each distinct crown.

2.  **Health Assessment Criteria:**
    -   **Healthy:** Deep green, full canopy, no visible discoloration.
    -   **Stressed:** Yellowing leaves, brown patches, thinning canopy, or pale green color.
    -   **Dead:** Gray/brown branches with no leaves, skeletal appearance, or completely brown canopy.

3.  **Coordinate Precision:**
    -   Output bounding boxes in [ymin, xmin, ymax, xmax] format.
    -   Coordinates must be normalized integers from 0 to 1000.
    -   Ensure boxes are tight around the visible crown.

**OUTPUT FORMAT:**
Return ONLY a valid JSON object. Do not include markdown formatting like ```json or ```.
Structure:
{
  "trees": [
    {
      "box": [ymin, xmin, ymax, xmax],
      "health": "healthy" | "stressed" | "dead",
      "confidence": float (0.0 to 1.0)
    }
  ],
  "lowGreenCoverDescription": "Brief analysis of the green cover density and any replanting recommendations."
}
"""

def get_vision_model():
    """Dynamically finds a working vision model to avoid deprecation errors."""
    try:
        models = client.models.list()
        vision_models = [m.id for m in models.data if 'vision' in m.id]
        
        # Prefer 90b (smartest), then 11b, then whatever is available
        for m in vision_models:
            if '90b' in m: return m
        if vision_models: return vision_models[0]
    except:
        pass
    return "llama-3.2-90b-vision-preview" 

@app.route('/')
def home():
    return "ReLeaf Backend (Optimized Groq AI) is Running! 🌲"

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_image():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.json
        if not data:
             return jsonify({"error": "No JSON received"}), 400
             
        img_data = data.get('image', '')
        if "data:image" not in img_data:
             base64_image = f"data:image/jpeg;base64,{img_data}"
        else:
             base64_image = img_data

        active_model = get_vision_model()
        print(f"Using model: {active_model}")

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image. Detect individual trees and assess health."},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ],
            model=active_model,
            temperature=0.1, # Low temperature for more factual/precise results
            max_tokens=2048,
            response_format={"type": "json_object"}, 
        )

        result = json.loads(chat_completion.choices[0].message.content)
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
