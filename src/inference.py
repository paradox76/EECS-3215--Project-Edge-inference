import sys
import os
import cv2
import numpy as np

TIDL_TOOLS_PATH = "/home/debian/object_detection_project/edgeai-tidl-tools/tools/AM68PA/tidl_tools"
sys.path.append(TIDL_TOOLS_PATH)
os.environ['LD_LIBRARY_PATH'] = TIDL_TOOLS_PATH + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import time
import json
from onnxruntime import InferenceSession

MODEL_PATH = "/home/debian/object_detection_project/models/yolox_s_lite_640x640_20220221_model.onnx"
OUTPUT_JSON_PATH = "/home/debian/object_detection_project/latest_detections.json"
OUTPUT_FRAME_PATH = "/home/debian/object_detection_project/latest_frame.jpg"

CAMERA_ID = 2 # always connect cam in bottom usb port
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def preprocess_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (INPUT_SIZE, INPUT_SIZE))
    
    input_tensor = np.transpose(resized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor

def postprocess_detections(outputs, frame_width, frame_height):
    """
    TI YOLOX has TWO outputs:
    outputs[0] = dets: (1, 1, 200, 5) -> [x1, y1, x2, y2, confidence]
    outputs[1] = labels: (1, 1, 1, 200) -> class IDs
    """
    detections = []
    
    dets = outputs[0][0][0]      # Shape: (200, 5)
    labels = outputs[1][0][0][0]  # Shape: (200,)
    
    for i in range(len(dets)):
        x1, y1, x2, y2, confidence = dets[i]
        class_id = int(labels[i])  # labels[i] is already a scalar
        
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            continue
        
        if class_id < 0 or class_id >= len(COCO_CLASSES):
            continue
        
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        
        scale_x = frame_width / INPUT_SIZE
        scale_y = frame_height / INPUT_SIZE
        
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        
        detections.append({
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'label': COCO_CLASSES[class_id],
            'class_id': class_id,
            'confidence': float(confidence)
        })
    
    return detections

def initialize_tidl_session():
    print("Initializing TIDL session...")
    print(f"Loading model from: {MODEL_PATH}")
    
    tidl_options = {
        'tidl_tools_path': TIDL_TOOLS_PATH,
        'artifacts_folder': '/opt/model_zoo/ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640/artifacts'
    }
    
    session = InferenceSession(
        MODEL_PATH,
        providers=['TIDLExecutionProvider', 'CPUExecutionProvider'],
        provider_options=[tidl_options, {}]
    )
    
    print("TIDL session initialized successfully")
    return session

def draw_detections(frame, detections):
    for detection in detections:
        x, y, w, h = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        label_text = f"{label} {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame, 
            (x, y - text_height - baseline - 5), 
            (x + text_width, y), 
            (0, 255, 0), 
            -1
        )
        cv2.putText(
            frame, 
            label_text, 
            (x, y - baseline - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1
        )
    
    return frame

def main():
    session = initialize_tidl_session()
    input_name = session.get_inputs()[0].name
    
    print(f"Opening camera {CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully")
    print("Starting inference loop...")
    
    frame_number = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_number += 1
            timestamp = time.time()
            
            input_tensor = preprocess_frame(frame)
            
            start_time = time.time()
            outputs = session.run(None, {input_name: input_tensor})
            inference_time = (time.time() - start_time) * 1000
            
            detections = postprocess_detections(outputs, FRAME_WIDTH, FRAME_HEIGHT)
            
            annotated_frame = draw_detections(frame.copy(), detections)
            
            cv2.imwrite(OUTPUT_FRAME_PATH, annotated_frame)
            
            result = {
                'timestamp': timestamp,
                'frame_number': frame_number,
                'frame_width': FRAME_WIDTH,
                'frame_height': FRAME_HEIGHT,
                'inference_time_ms': round(inference_time, 2),
                'num_detections': len(detections),
                'detections': detections
            }
            
            with open(OUTPUT_JSON_PATH, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Frame {frame_number}: {len(detections)} detections, {inference_time:.1f}ms")
            
    except KeyboardInterrupt:
        print("\nStopping inference...")
    
    finally:
        cap.release()
        print("Camera released")

if __name__ == "__main__":
    main()