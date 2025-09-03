
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO

# Set template_folder to frontend for deployment
app = Flask(__name__, template_folder='../frontend')

# Load YOLOv8-X model (pre-trained on COCO dataset)
model = YOLO('yolov8x.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    results = model(img)
    objects = []
    for r in results:
        boxes = r.boxes
        names = r.names
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = names[cls]
            objects.append({
                'label': label,
                'confidence': conf,
                'box': [int(x1), int(y1), int(x2), int(y2)]
            })
    return jsonify(objects)

if __name__ == '__main__':
    app.run(debug=True)
