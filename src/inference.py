import sys
import os
import cv2
import numpy as np

TIDL_TOOLS_PATH = "/home/beagle/object_detection_project/edgeai-tidl-tools/tools/AM68PA/tidl_tools"
sys.path.append(TIDL_TOOLS_PATH)

os.environ['LD_LIBRARY_PATH'] = TIDL_TOOLS_PATH + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import time
import json
from onnxruntime import InferenceSession

MODEL_PATH = "/home/beagle/object_detection_project/models/yolo11n.onnx"
OUTPUT_JSON_PATH = "/home/beagle/object_detection_project/latest_detections.json"
OUTPUT_FRAME_PATH = "/home/beagle/object_detection_project/latest_frame.jpg"

CAMERA_ID = 0  # depends on the number of cams connected
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

INPUT_SIZE = 640  # model expects this size
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to consider a detection valid
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
    
    normalized = resized.astype(np.float32) / 255.0
    
    input_tensor = np.transpose(normalized, (2, 0, 1))  # (3, 640, 640)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, 640, 640)
    
    return input_tensor



def postprocess_detections(output, frame_width, frame_height):
  
    detections = []
    
    
    output = output[0]
    
    output = np.transpose(output)
    
    for detection in output:
        x_center, y_center, w, h = detection[:4]
        
        class_scores = detection[4:]
        
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence >= CONFIDENCE_THRESHOLD:
            x1 = int((x_center - w/2) * frame_width / INPUT_SIZE)
            y1 = int((y_center - h/2) * frame_height / INPUT_SIZE)
            x2 = int((x_center + w/2) * frame_width / INPUT_SIZE)
            y2 = int((y_center + h/2) * frame_height / INPUT_SIZE)
            
            detections.append({
                'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                'label': COCO_CLASSES[class_id],
                'class_id': int(class_id),
                'confidence': float(confidence)
            })
    
    return detections


def initialize_tidl_session():
    """
    Create ONNX Runtime session with TIDL execution provider.
    This compiles the model for hardware acceleration.
    """
    print("Initializing TIDL session...")
    print(f"Loading model from: {MODEL_PATH}")
    
    # TIDL execution provider options
    tidl_options = {
        'tidl_tools_path': TIDL_TOOLS_PATH,
        'artifacts_folder': '/home/beagle/object_detection_project/tidl_artifacts'
    }
    
    # Create session with TIDL provider
    session = InferenceSession(
        MODEL_PATH,
        providers=['TIDLExecutionProvider', 'CPUExecutionProvider'],
        provider_options=[tidl_options, {}]
    )
    
    print("TIDL session initialized successfully")
    return session

def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on the frame.
    Returns the annotated frame.
    """
    for detection in detections:
        x, y, w, h = detection['bbox']
        label = detection['label']
        confidence = detection['confidence']
        
        # Draw bounding box (green)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label with background
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

# main loop below


def main():
    """
    Main inference loop: capture frames, run detection, write results to JSON.
    """
    # Initialize TIDL session
    session = initialize_tidl_session()
    
    # Get model input name
    input_name = session.get_inputs()[0].name
    
    # Open camera
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
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_number += 1
            timestamp = time.time()
            
            # Preprocess frame
            input_tensor = preprocess_frame(frame)
            
            # Run inference
            start_time = time.time()
            outputs = session.run(None, {input_name: input_tensor})
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Postprocess to get detections
            detections = postprocess_detections(outputs[0], FRAME_WIDTH, FRAME_HEIGHT)
            
            # Draw detections on frame
            annotated_frame = draw_detections(frame.copy(), detections)
            
            # Save annotated frame
            cv2.imwrite(OUTPUT_FRAME_PATH, annotated_frame)
            
            # Create output JSON
            result = {
                'timestamp': timestamp,
                'frame_number': frame_number,
                'frame_width': FRAME_WIDTH,
                'frame_height': FRAME_HEIGHT,
                'inference_time_ms': round(inference_time, 2),
                'num_detections': len(detections),
                'detections': detections
            }
            
            # Write to JSON file
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


