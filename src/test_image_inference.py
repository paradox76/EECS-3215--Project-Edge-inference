import sys
import os
import cv2
import numpy as np
import time
import json

TIDL_TOOLS_PATH = "/home/debian/object_detection_project/edgeai-tidl-tools/tools/AM68PA/tidl_tools"
sys.path.append(TIDL_TOOLS_PATH)
os.environ['LD_LIBRARY_PATH'] = TIDL_TOOLS_PATH + ':' + os.environ.get('LD_LIBRARY_PATH', '')

from onnxruntime import InferenceSession
from inference import preprocess_frame, postprocess_detections, draw_detections, initialize_tidl_session

TEST_IMAGE_PATH = "/home/debian/object_detection_project/test_images/test2.jpg"
OUTPUT_PATH = "/home/debian/object_detection_project/test2_output.jpg"

def main():
    print(f"Loading image: {TEST_IMAGE_PATH}")
    frame = cv2.imread(TEST_IMAGE_PATH)
    
    if frame is None:
        print("Error: Could not load image")
        return
    
    h, w = frame.shape[:2]
    print(f"Image size: {w}x{h}")
    
    session = initialize_tidl_session()
    input_name = session.get_inputs()[0].name
    
    input_tensor = preprocess_frame(frame)
    
    print("Running inference...")
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = (time.time() - start) * 1000

    # DEBUG: Check output structure
    print(f"\nDEBUG - Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output[{i}] shape: {out.shape}")
        
    detections = postprocess_detections(outputs, w, h)
    
    detections = postprocess_detections(outputs, w, h)
    
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