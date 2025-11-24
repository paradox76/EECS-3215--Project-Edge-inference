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

CAMERA_ID = 0  # Usually 0 for first USB webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

INPUT_SIZE = 640  # YOLOv11 expects 640x640 input
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
    """
    Prepare camera frame for YOLO model input.
    Takes BGR frame from OpenCV, converts and resizes for YOLO.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    resized = cv2.resize(rgb_frame, (INPUT_SIZE, INPUT_SIZE))
    
    normalized = resized.astype(np.float32) / 255.0
    
    input_tensor = np.transpose(normalized, (2, 0, 1))  # (3, 640, 640)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 3, 640, 640)
    
    return input_tensor