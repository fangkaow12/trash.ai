import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0 # โมเดลที่ใช้เทรน
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

app = Flask(__name__)

INPUT_SHAPE = (640, 640) # ขนาดที่โมเดลของคุณต้องการ
# ชื่อคลาสที่ใช้ทำนายต้องเรียงลำดับตามการเทรนโมเดล
CLASS_NAMES = ['general_trash', 'glass', 'hazardous_waste', 'Plastic_bottle'] 
NUM_CLASSES = len(CLASS_NAMES)

# โหลด Weights ตามโมเดลที่ใช้เทรน คือ EfficientNetB0
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3))

# สร้าง Classification Head 
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# รวม Base Model และ Head (full model)
model = Model(inputs=base_model.input, outputs=predictions)

# โหลด Weights 
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
model.load_weights(WEIGHTS_PATH)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    # ถ้าไม่มีโฟลเดอร์ images ก็สร้างซะเลย
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(images_dir, exist_ok=True)
    image_path = os.path.join(images_dir, imagefile.filename)
    imagefile.save(image_path)

    # แก้ไข Preprocessing ให้เหมาะกับโมเดล
    image = load_img(image_path, target_size=INPUT_SHAPE)
    image = img_to_array(image)
    image = image / 255.0  # Normalize ค่า pixel ให้อยู่ระหว่าง 0-1
    image = np.expand_dims(image, axis=0) # เพิ่ม dimension สำหรับ batch size -> (1, 224, 224, 3)

    # ทำนายผลและแปลผลลัพธ์
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    classification = f'{predicted_class_name} ({confidence:.2f}%)'

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)