import base64
import os
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------- Config ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.h5")

# คลาสที่ใช้ทำนาย (เรียงตามตอนเทรน)
CLASS_NAMES = ['general_trash', 'glass', 'hazardous_waste', 'Plastic_bottle']
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (640, 640)  # ขนาดที่ EfficientNetB0 ใช้

# ---------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/predict_api": {"origins": "*"}})  # เปิดให้เรียก API ได้จากภายนอก

# ---------------- โหลดโมเดล ----------------
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(MODEL_PATH)

# ---------------- ฟังก์ชันเตรียมภาพ ----------------
def preprocess_pil(img: Image.Image):
    """ใช้กับ API (รับ base64)"""
    img = img.convert("RGB")
    img = img.resize(INPUT_SHAPE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def preprocess_file(image_path: str):
    """ใช้กับหน้าเว็บ (รับไฟล์จริง)"""
    image = load_img(image_path, target_size=INPUT_SHAPE)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    """หน้าเว็บหลัก"""
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict_from_form():
    """รับภาพจากหน้าเว็บและแสดงผล"""
    try:
        imagefile = request.files.get("imagefile")
        if not imagefile:
            return render_template("index.html", prediction="❌ กรุณาเลือกภาพก่อน")

        # เก็บไฟล์ในโฟลเดอร์ images/
        images_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(images_dir, exist_ok=True)
        image_path = os.path.join(images_dir, imagefile.filename)
        imagefile.save(image_path)

        # เตรียมข้อมูลและทำนาย
        x = preprocess_file(image_path)
        preds = model.predict(x)
        probs = preds[0]
        predicted_index = int(np.argmax(probs))
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(probs[predicted_index]) * 100

        result_text = f"{predicted_label} ({confidence:.2f}%)"

        return render_template("index.html", prediction=result_text)

    except Exception as e:
        return render_template("index.html", prediction=f"เกิดข้อผิดพลาด: {str(e)}")

@app.route("/predict_api", methods=["POST", "OPTIONS"])
def predict_api():
    """API สำหรับรับ base64 แล้วตอบ JSON"""
    try:
        data = request.get_json(silent=True)
        if not data or "image_base64" not in data:
            return jsonify({"error": "No image provided"}), 400

        b64 = data["image_base64"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]

        img = Image.open(BytesIO(base64.b64decode(b64)))
        x = preprocess_pil(img)

        preds = model.predict(x)
        probs = preds[0]
        predicted_index = int(np.argmax(probs))
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(probs[predicted_index])
        probs_dict = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}

        return jsonify({
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probs": probs_dict
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1300, debug=True)