# Driift

A smart camera gimbal system that automatically tracks a person in real time using computer vision.

Developed by **Naman & Manan**.

---

## Overview

Driift solves the issue of static cameras during presentations, lectures, or solo filming. A webcam streams live video to a laptop, where a Python script uses **YOLOv8** to track the user. The system calculates the subject's offset from the center of the frame and communicates via serial to an **Arduino**, which adjusts pan and tilt motors to keep the subject perfectly framed.

### Key Features
* **Real-Time Tracking:** Powered by UltraLytics YOLOv8 for accurate person detection.
* **Gesture Control:** Built-in MediaPipe hand tracking allows users to pause/resume tracking with a fist gesture.
* **Hardware Integration:** Smooth Python-to-Arduino serial data communication.
* **Dynamic Correction:** Continuous X and Y offset calculation for fluid motor adjustments.

---

## Use Cases
* **Education:** Presenting at a whiteboard without walking out of frame.
* **Content Creation:** Recording video essays, demos, or vlogs without an operator.
* **Remote Work:** Keeping dynamic and engaged during video conferences.

---

## Inspiration
* [DJI Auto Tracking Gimbals](https://www.dji.com)
* [OBS AI Auto Framing](https://obsproject.com)
* [OpenCV Computer Vision Tools](https://opencv.org)
* [NVIDIA Real-Time Detection](https://developer.nvidia.com)

---

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the Python backend:
   ```bash
   py main.py