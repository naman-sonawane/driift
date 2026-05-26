import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt') 

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# State variables for the tracking toggle
is_paused = False
fist_cooldown = 0  # Prevents rapid flickering between pause/resume

def is_fist(hand_landmarks):
    """Detects a fist by checking if fingers are curled into the palm."""
    # Tip and PIP joint indices for: Index, Middle, Ring, Pinky
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    
    # Count how many fingers are folded down
    folded_fingers = 0
    for tip, pip in zip(tips, pip_joints):
        # In MediaPipe, Y increases downwards. 
        # Tip Y > PIP Y means the finger is curled down.
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            folded_fingers += 1
            
    # If 4 fingers are folded, consider it a fist
    return folded_fingers == 4

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    # Process hand gesture logic
    if hand_results.multi_hand_landmarks and fist_cooldown == 0:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if is_fist(hand_landmarks):
                is_paused = not is_paused  # Toggle the state
                fist_cooldown = 15          # Wait ~15 frames before toggling again
                
                if is_paused:
                    print("PAUSED")
                else:
                    print("RESUME")
                break

    # Decrement cooldown timer
    if fist_cooldown > 0:
        fist_cooldown -= 1

    # Visual overlay for state feedback
    state_text = "PAUSED" if is_paused else "PLAYING"
    text_color = (0, 0, 255) if is_paused else (0, 255, 0)
    cv2.putText(frame, f"STATUS: {state_text}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # ONLY TRACK YOLO POSITIONS IF NOT PAUSED
    if not is_paused:
        # HIGH CONFIDENCE TRACKING:
        results = model.track(
            frame, 
            persist=True, 
            classes=0, 
            conf=0.8, 
            iou=0.5
        ) 

        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center_x, center_y = int(x), int(y)

                # Visual overlay
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID {track_id} ({center_x}, {center_y})", 
                            (center_x - 50, center_y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Person Tracker (Filtered)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()