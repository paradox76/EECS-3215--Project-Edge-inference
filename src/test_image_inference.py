import sys
import os
import cv2
import numpy as np
import time
import json

# Add TIDL paths
TIDL_TOOLS_PATH = "/home/beagle/object_detection_project/edgeai-tidl-tools/tools/AM68PA/tidl_tools"
sys.path.append(TIDL_TOOLS_PATH)
os.environ['LD_LIBRARY_PATH'] = TIDL_TOOLS_PATH + ':' + os.environ.get('LD_LIBRARY_PATH', '')

from onnxruntime import InferenceSession

# Import functions from inference.py
from inference import preprocess_frame, postprocess_detections, draw_detections, initialize_tidl_session

# Test image path
TEST_IMAGE_PATH = "/home/beagle/object_detection_project/test_images/test.jpg"
OUTPUT_PATH = "/home/beagle/object_detection_project/test_output.jpg"

def main():
    # Load test image
    print(f"Loading image: {TEST_IMAGE_PATH}")
    frame = cv2.imread(TEST_IMAGE_PATH)
    
    if frame is None:
        print("Error: Could not load image")
        return
    
    h, w = frame.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Initialize TIDL
    session = initialize_tidl_session()
    input_name = session.get_inputs()[0].name
    
    # Preprocess
    input_tensor = preprocess_frame(frame)
    
    # Run inference
    print("Running inference...")
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = (time.time() - start) * 1000
    
    # Postprocess
    detections = postprocess_detections(outputs[0], w, h)
    
    # Draw and save
    annotated = draw_detections(frame.copy(), detections)
    cv2.imwrite(OUTPUT_PATH, annotated)
    
    print(f"\nResults:")
    print(f"Inference time: {inference_time:.1f}ms")
    print(f"Detections: {len(detections)}")
    for det in detections:
        print(f"  - {det['label']}: {det['confidence']:.2f}")
    print(f"\nOutput saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
