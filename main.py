import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Load the YOLO pose model (person boxes + body keypoints)
model = YOLO('yolov8n-pose.pt')

# Initialize MediaPipe Tasks Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure the landmarker for video stream mode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)

# State variables for the tracking toggle
is_paused = False
fist_cooldown = 0  # Prevents rapid flickering between pause/resume
fist_hold_frames = 0  # Requires a stable fist for multiple frames
open_hold_frames = 0  # Requires a stable open hand for resume

FULL_HAND_MIN_SPAN = 0.25
FIST_HOLD_FRAMES = 4
OPEN_HOLD_FRAMES = 3
LOOKING_SCORE_THRESHOLD = 0.45
SUBJECT_DEADBAND_PX = 12   # Ignore tiny jitter under this pixel distance.
SUBJECT_SMOOTHING_ALPHA = 0.25  # Lower value = smoother, less responsive.
SUBJECT_REACQUIRE_PX = 80   # Snap quickly if target jumps a large distance.
GIMBAL_DEADBAND_RATIO = 0.03  # Ignore tiny center errors (<3% of frame half-size).
GIMBAL_GAIN_X = 0.8
GIMBAL_GAIN_Y = 0.8
GIMBAL_MAX_OUTPUT = 1.0  # Output command range is [-1.0, 1.0].

subject_track_id = None
subject_stable_center = None

def is_fist(hand_landmarks):
    """Detects a fist by checking if fingers are curled into the palm."""
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    
    folded_fingers = 0
    for tip, pip in zip(tips, pip_joints):
        if hand_landmarks[tip].y > hand_landmarks[pip].y:
            folded_fingers += 1
            
    return folded_fingers == 4

def is_open_hand(hand_landmarks):
    """Detects an open hand by checking if fingers are extended."""
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]

    extended_fingers = 0
    for tip, pip in zip(tips, pip_joints):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            extended_fingers += 1

    return extended_fingers >= 4

def is_full_hand_visible(hand_landmarks):
    """Checks that the hand is fully in frame and large enough to be reliable."""
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    # All landmarks should be within view with a tiny edge margin.
    margin = 0.03
    if min(xs) < margin or max(xs) > 1 - margin:
        return False
    if min(ys) < margin or max(ys) > 1 - margin:
        return False

    # Hand should occupy enough image area to avoid tiny false positives.
    hand_width = max(xs) - min(xs)
    hand_height = max(ys) - min(ys)
    return max(hand_width, hand_height) >= FULL_HAND_MIN_SPAN

def get_facing_camera_score(keypoints):
    """
    Returns a 0..1 score for whether a person appears to face the camera.
    Uses YOLO pose keypoints: nose(0), left_eye(1), right_eye(2), left_shoulder(5), right_shoulder(6).
    """
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    if np.any(nose <= 0) or np.any(left_eye <= 0) or np.any(right_eye <= 0):
        return 0.0
    if np.any(left_shoulder <= 0) or np.any(right_shoulder <= 0):
        return 0.0

    inter_eye = abs(right_eye[0] - left_eye[0])
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    if inter_eye < 5 or shoulder_width < 10:
        return 0.0

    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_level_diff = abs(left_eye[1] - right_eye[1]) / inter_eye
    nose_eye_offset = abs(nose[0] - eye_center_x) / inter_eye
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    nose_shoulder_offset = abs(nose[0] - shoulder_center_x) / shoulder_width

    # Frontal face heuristics: balanced eyes and centered nose.
    eye_balance = max(0.0, 1.0 - min(eye_level_diff / 0.35, 1.0))
    nose_between_eyes = max(0.0, 1.0 - min(nose_eye_offset / 0.55, 1.0))
    shoulder_alignment = max(0.0, 1.0 - min(nose_shoulder_offset / 0.65, 1.0))

    return 0.45 * nose_between_eyes + 0.35 * eye_balance + 0.20 * shoulder_alignment

def stabilize_subject_center(previous_center, detected_center):
    """Applies deadband + smoothing to reduce subject-point vibration."""
    if previous_center is None:
        return detected_center

    prev_x, prev_y = previous_center
    det_x, det_y = detected_center
    dx = det_x - prev_x
    dy = det_y - prev_y
    dist = float(np.hypot(dx, dy))

    # Reject micro-movements so the gimbal does not chatter.
    if dist < SUBJECT_DEADBAND_PX:
        return previous_center

    # If subject position jumps a lot, reacquire quickly.
    if dist > SUBJECT_REACQUIRE_PX:
        return detected_center

    smooth_x = int(prev_x + SUBJECT_SMOOTHING_ALPHA * dx)
    smooth_y = int(prev_y + SUBJECT_SMOOTHING_ALPHA * dy)
    return (smooth_x, smooth_y)

def compute_gimbal_output(subject_center, frame_w, frame_h):
    """
    Returns normalized gimbal movement commands:
    - move_x: negative=left, positive=right
    - move_y: negative=up, positive=down
    """
    if subject_center is None:
        return 0.0, 0.0

    center_x, center_y = subject_center
    frame_center_x = frame_w / 2.0
    frame_center_y = frame_h / 2.0
    half_w = max(frame_w / 2.0, 1.0)
    half_h = max(frame_h / 2.0, 1.0)

    err_x = (center_x - frame_center_x) / half_w
    err_y = (center_y - frame_center_y) / half_h

    if abs(err_x) < GIMBAL_DEADBAND_RATIO:
        err_x = 0.0
    if abs(err_y) < GIMBAL_DEADBAND_RATIO:
        err_y = 0.0

    move_x = float(np.clip(err_x * GIMBAL_GAIN_X, -GIMBAL_MAX_OUTPUT, GIMBAL_MAX_OUTPUT))
    move_y = float(np.clip(err_y * GIMBAL_GAIN_Y, -GIMBAL_MAX_OUTPUT, GIMBAL_MAX_OUTPUT))
    return move_x, move_y

def get_nose_point(keypoints):
    """Returns (x, y) for nose keypoint if valid, else None."""
    if keypoints is None or len(keypoints) == 0:
        return None
    nose = keypoints[0]
    if np.any(nose <= 0):
        return None
    return int(nose[0]), int(nose[1])

def get_torso_center_point(keypoints):
    """
    Returns (x, y) for the torso center using pose keypoints if valid, else None.
    Uses shoulders (5,6) and hips (11,12) when available; falls back to shoulders-only.
    """
    if keypoints is None or len(keypoints) == 0:
        return None

    def valid_xy(kp):
        return kp is not None and not np.any(kp <= 0)

    left_shoulder = keypoints[5] if len(keypoints) > 6 else None
    right_shoulder = keypoints[6] if len(keypoints) > 6 else None
    left_hip = keypoints[11] if len(keypoints) > 12 else None
    right_hip = keypoints[12] if len(keypoints) > 12 else None

    points = []
    for kp in (left_shoulder, right_shoulder, left_hip, right_hip):
        if kp is not None and valid_xy(kp):
            points.append(kp)

    if len(points) >= 3:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    shoulder_points = []
    for kp in (left_shoulder, right_shoulder):
        if kp is not None and valid_xy(kp):
            shoulder_points.append(kp)
    if len(shoulder_points) == 2:
        x = int((shoulder_points[0][0] + shoulder_points[1][0]) / 2.0)
        y = int((shoulder_points[0][1] + shoulder_points[1][1]) / 2.0)
        return x, y

    return None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0:
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    hand_results = landmarker.detect_for_video(mp_image, timestamp_ms)

    full_hand_detected = False
    if hand_results.hand_landmarks:
        for hand_landmarks in hand_results.hand_landmarks:
            if not is_full_hand_visible(hand_landmarks):
                continue

            full_hand_detected = True
            if is_paused:
                fist_hold_frames = 0
                if is_open_hand(hand_landmarks):
                    open_hold_frames += 1
                    if fist_cooldown == 0 and open_hold_frames >= OPEN_HOLD_FRAMES:
                        is_paused = False
                        fist_cooldown = 20
                        open_hold_frames = 0
                        print("RESUME")
                else:
                    open_hold_frames = 0
            else:
                open_hold_frames = 0
                if is_fist(hand_landmarks):
                    fist_hold_frames += 1
                    if fist_cooldown == 0 and fist_hold_frames >= FIST_HOLD_FRAMES:
                        is_paused = True
                        fist_cooldown = 20
                        fist_hold_frames = 0
                        print("PAUSED")
                else:
                    fist_hold_frames = 0
            break

    if not full_hand_detected:
        fist_hold_frames = 0
        open_hold_frames = 0

    if fist_cooldown > 0:
        fist_cooldown -= 1

    state_text = "PAUSED" if is_paused else "PLAYING"
    text_color = (0, 0, 255) if is_paused else (0, 255, 0)
    cv2.putText(frame, f"STATUS: {state_text}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    coordinate_lines = []
    gimbal_output = (0.0, 0.0)
    frame_h, frame_w = frame.shape[:2]

    if not is_paused:
        results = model.track(frame, persist=True, classes=0, conf=0.8, iou=0.5, verbose=False) 

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
            keypoints_xy = None
            if results[0].keypoints is not None and results[0].keypoints.xy is not None:
                keypoints_xy = results[0].keypoints.xy.cpu().numpy()

            best_subject_id = None
            best_subject_score = -1.0
            candidate_scores = {}

            for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x, y, w, h = box
                center_x, center_y = int(x), int(y)
                nose_x, nose_y = center_x, center_y

                facing_score = 0.0
                if keypoints_xy is not None and idx < len(keypoints_xy):
                    facing_score = get_facing_camera_score(keypoints_xy[idx])
                    nose_point = get_nose_point(keypoints_xy[idx])
                    if nose_point is not None:
                        nose_x, nose_y = nose_point

                area_score = min((w * h) / float(frame_w * frame_h * 0.35), 1.0)
                center_dist_x = abs(nose_x - (frame_w // 2)) / max(frame_w // 2, 1)
                center_dist_y = abs(nose_y - (frame_h // 2)) / max(frame_h // 2, 1)
                center_score = max(0.0, 1.0 - 0.6 * center_dist_x - 0.4 * center_dist_y)

                subject_score = 0.65 * facing_score + 0.20 * area_score + 0.15 * center_score
                candidate_scores[track_id] = (subject_score, facing_score)

                if facing_score >= LOOKING_SCORE_THRESHOLD and subject_score > best_subject_score:
                    best_subject_score = subject_score
                    best_subject_id = track_id

            # Keep the existing subject if still a valid facing candidate to reduce jumping.
            if subject_track_id in candidate_scores:
                current_subject_score, current_facing_score = candidate_scores[subject_track_id]
                if current_facing_score >= LOOKING_SCORE_THRESHOLD and current_subject_score >= best_subject_score - 0.07:
                    best_subject_id = subject_track_id
                    best_subject_score = current_subject_score

            previous_subject_id = subject_track_id
            subject_track_id = best_subject_id
            if subject_track_id != previous_subject_id:
                subject_stable_center = None

            for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x, y, w, h = box
                center_x, center_y = int(x), int(y)
                facing_score = 0.0
                nose_point = None
                torso_point = None
                if keypoints_xy is not None and idx < len(keypoints_xy):
                    facing_score = get_facing_camera_score(keypoints_xy[idx])
                    nose_point = get_nose_point(keypoints_xy[idx])
                    torso_point = get_torso_center_point(keypoints_xy[idx])

                is_subject = track_id == subject_track_id
                color = (0, 255, 255) if is_subject else (0, 255, 0)
                label_prefix = "SUBJECT" if is_subject else f"ID {track_id}"

                display_x, display_y = center_x, center_y
                if is_subject:
                    if nose_point is not None:
                        display_x, display_y = nose_point
                    else:
                        display_x, display_y = center_x, center_y
                    subject_stable_center = (display_x, display_y)
                    gimbal_output = compute_gimbal_output((display_x, display_y), frame_w, frame_h)
                else:
                    if torso_point is not None:
                        display_x, display_y = torso_point

                cv2.circle(frame, (display_x, display_y), 5, color, -1)
                cv2.putText(frame, f"{label_prefix} ({display_x}, {display_y})",
                            (display_x - 70, display_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                coordinate_lines.append(
                    f"{label_prefix}: ({display_x}, {display_y}) facing={facing_score:.2f}"
                )
        else:
            subject_track_id = None
            subject_stable_center = None
            gimbal_output = (0.0, 0.0)
    else:
        subject_track_id = None
        subject_stable_center = None
        gimbal_output = (0.0, 0.0)

    move_x, move_y = gimbal_output
    cv2.putText(frame, f"GIMBAL x:{move_x:+.2f} y:{move_y:+.2f}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if is_paused:
        print("TRACKING PAUSED | GIMBAL x:+0.00 y:+0.00")
    elif coordinate_lines:
        print(" | ".join(coordinate_lines) + f" | GIMBAL x:{move_x:+.2f} y:{move_y:+.2f}")
    else:
        print("No tracked coordinates | GIMBAL x:+0.00 y:+0.00")

    cv2.imshow("Person Tracker (Filtered)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
