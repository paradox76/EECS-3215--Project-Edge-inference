# EECS-3215--Project-Edge-inference

# Real-Time Object Detection with TIDL Hardware Acceleration

Edge AI object detection system running on BeagleBone AI-64 with TI Deep Learning (TIDL) hardware acceleration.

## Performance
- **80 FPS** real-time inference (12.5ms average)
- **289 neural network layers** offloaded to C7x DSP
- **20-30x speedup** vs CPU-only inference

## Hardware Requirements
- BeagleBone AI-64
- USB Webcam
- 5V/3A power adapter (barrel jack recommended for stable power during inference)

## Model
- **YOLOX-S-Lite** (TI optimized, COCO 80 classes)
- Located in `/models/yolox_s_lite_640x640_20220221_model.onnx`
- Pre-compiled TIDL artifacts from TI model zoo

## Output Files
- `latest_detections.json` - Detection results (labels, bboxes, confidence, timing)
- `latest_frame.jpg` - Annotated frame with bounding boxes

## Setup
1. Install TIDL tools:
```bash
   cd /home/debian/object_detection_project
   git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
```

2. Run inference:
```bash
   cd src
   sudo python3 inference.py
```

## Dependencies
- OpenCV 4.5.1
- NumPy 1.19.5
- onnxruntime-tidl 1.7.0

## Notes
- YOLOv11n attempted but incompatible with TIDL (opset 22 > supported opset 19)
- Camera device: `/dev/video2` (modify `CAMERA_ID` if different)
## Pre-installed on BeagleBone AI-64:
- onnxruntime-tidl 1.7.0
- TIDL runtime libraries