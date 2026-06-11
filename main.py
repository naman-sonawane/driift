import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from ultralytics import YOLO
import sys
import time
import serial
import serial.tools.list_ports

# Load the YOLO pose model (person boxes + body keypoints)
model = YOLO('yolov8n-pose.pt')
hand_detector = HandDetector(detectionCon=0.7, maxHands=1)

# ── Arduino serial (115200 baud — matches main/main.ino) ─────────────────────
arduino = None
connection_lost = False
last_heartbeat = 0.0
ARDUINO_BAUD = 115200
HEARTBEAT_INTERVAL = 5.0


def find_arduino_port():
    """Return the first port that looks like an Arduino."""
    for port in serial.tools.list_ports.comports():
        desc = port.description or ""
        if any(tag in desc for tag in ("Arduino", "CH340", "USB Serial")):
            return port.device
    return None


def connect_to_arduino():
    global arduino, connection_lost, last_heartbeat
    arduino_port = find_arduino_port() or "COM3"
    try:
        print(f"Connecting to Arduino on {arduino_port}...")
        arduino = serial.Serial(arduino_port, ARDUINO_BAUD, timeout=1)
        time.sleep(2)  # Wait for Arduino reset after USB open
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if arduino.in_waiting:
                msg = arduino.readline().decode(errors="replace").strip()
                if msg:
                    print(f"Arduino: {msg}")
                if msg == "DRIIFT_READY":
                    connection_lost = False
                    last_heartbeat = time.time()
                    print("Arduino connected — DRIIFT hardware online.")
                    return True
            time.sleep(0.05)
        print("Warning: Arduino did not send DRIIFT_READY; continuing anyway.")
        connection_lost = False
        last_heartbeat = time.time()
        return True
    except Exception as exc:
        print(f"Failed to connect to Arduino: {exc}")
        arduino = None
        return False


def drain_arduino_inbox():
    """Print any pending lines from the Arduino (non-blocking)."""
    if arduino is None or not arduino.is_open:
        return
    try:
        while arduino.in_waiting:
            msg = arduino.readline().decode(errors="replace").strip()
            if msg:
                print(f"Arduino: {msg}")
    except Exception as exc:
        print(f"Serial read error: {exc}")


def check_arduino_connection():
    global last_heartbeat
    if arduino is None or not arduino.is_open:
        return False
    try:
        if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
            arduino.write(b"PING\n")
            last_heartbeat = time.time()
        return True
    except Exception:
        return False


def handle_connection_loss():
    global connection_lost
    if not connection_lost:
        connection_lost = True
        print("Arduino disconnected — servos will hold last position.")


def reconnect_arduino():
    global arduino
    print("Attempting to reconnect to Arduino...")
    if arduino is not None:
        try:
            arduino.close()
        except Exception:
            pass
        arduino = None
    return connect_to_arduino()


def send_arduino_command(command):
    """Send a newline-terminated command to the Arduino."""
    global last_heartbeat
    if arduino is None or connection_lost:
        return False
    try:
        arduino.write(f"{command}\n".encode())
        last_heartbeat = time.time()
        return True
    except Exception as exc:
        print(f"Serial write error: {exc}")
        handle_connection_loss()
        return False



def direction_for_arduino(subject_point, frame_w, active):
    """
    Map subject position to a simple mount command for the Arduino.
    Returns LEFT | RIGHT | SAFE (deadzone / inactive).
    """
    if not active or subject_point is None:
        return "SAFE"

    off_x = subject_point[0] - frame_w // 2
    if abs(off_x) <= TRACK_DEADZONE_PX:
        return "SAFE"
    return "RIGHT" if off_x > 0 else "LEFT"


def send_tracking_to_arduino(subject_point, frame_w, frame_h, active):
    """Push LEFT | RIGHT | SAFE to the Arduino every frame (main.ino)."""
    send_arduino_command(direction_for_arduino(subject_point, frame_w, active))

def open_camera(preferred_indices=(1, 0, 2, 3)):
    """Try camera indices in order and return the first working capture."""
    for cam_idx in preferred_indices:
        cap_candidate = cv2.VideoCapture(cam_idx)
        if cap_candidate is not None and cap_candidate.isOpened():
            print(f"Using camera index {cam_idx}")
            return cap_candidate
        if cap_candidate is not None:
            cap_candidate.release()
    return None

cap = open_camera()
if cap is None:
    print("ERROR: Could not open any camera device. Try connecting a camera or changing indices.")
    sys.exit(1)

if not connect_to_arduino():
    print("ERROR: Could not connect to Arduino. Check USB cable and upload main/main.ino.")
    sys.exit(1)

print("Controls:")
print("  Fist  -> PAUSE tracking")
print("  Palm  -> PLAY / resume tracking")
print("  Peace -> toggle gimbal hold")
print("  P key -> toggle pause")
print("  R key -> re-center servos (RESET)")
print("  Q key -> quit\n")

DISPLAY_WINDOW = "Person Tracker (Filtered)"
_display_window_ready = False

def _get_window_client_size(fallback_w, fallback_h):
    """Actual window client area (OpenCV getWindowImageRect can report image size, not window)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            hwnd = ctypes.windll.user32.FindWindowW(None, DISPLAY_WINDOW)
            if hwnd:
                rect = wintypes.RECT()
                if ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w > 0 and h > 0:
                        return w, h
        except Exception:
            pass

    try:
        _, _, w, h = cv2.getWindowImageRect(DISPLAY_WINDOW)
        if w > 0 and h > 0:
            return w, h
    except cv2.error:
        pass
    return fallback_w, fallback_h

def show_frame(frame):
    """Letterbox frame into the window so aspect ratio is never stretched."""
    global _display_window_ready
    if not _display_window_ready:
        cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DISPLAY_WINDOW, frame.shape[1], frame.shape[0])
        _display_window_ready = True

    fh, fw = frame.shape[:2]
    win_w, win_h = _get_window_client_size(fw, fh)

    scale = min(win_w / fw, win_h / fh)
    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))
    scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if new_w == win_w and new_h == win_h:
        display = scaled
    else:
        display = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x0 = (win_w - new_w) // 2
        y0 = (win_h - new_h) // 2
        display[y0 : y0 + new_h, x0 : x0 + new_w] = scaled

    cv2.imshow(DISPLAY_WINDOW, display)

# State variables for the tracking toggle
is_paused = False
gimbal_frozen = False
frozen_gimbal_output = (0.0, 0.0)
gimbal_freeze_capture = False  # Latch current gimbal output on next frame after peace sign.

# Gesture debounce: must be stable for N consecutive frames before triggering
GESTURE_HOLD = 8
gesture_buffer = []
last_triggered_gesture = None
LOOKING_SCORE_THRESHOLD = 0.45
SUBJECT_DEADBAND_PX = 25   # Ignore tiny jitter under this pixel distance.
SUBJECT_SMOOTHING_ALPHA = 0.25  # Lower value = smoother, less responsive.
SUBJECT_REACQUIRE_PX = 80   # Snap quickly if target jumps a large distance.
TRACK_DEADZONE_PX = 40  # Subject within this many pixels of center -> SAFE
GIMBAL_DEADBAND_RATIO = 0.03  # Ignore tiny center errors (<3% of frame half-size).
GIMBAL_GAIN_X = 0.8
GIMBAL_GAIN_Y = 0.8
GIMBAL_MAX_OUTPUT = 1.0  # Output command range is [-1.0, 1.0].

subject_track_id = None
subject_stable_center = None

# HUD / telemetry state
fps_value = 0.0
fps_frame_count = 0
fps_last_tick = time.perf_counter()
fake_battery_pct = 92.0
battery_last_tick = time.perf_counter()
HUD_ACCENT = (0, 255, 255)
HUD_DIM = (170, 170, 170)
HUD_WARN = (0, 140, 255)
HUD_ALERT = (0, 0, 255)
STARTUP_SCAN_DURATION = 2.75
STARTUP_IDENTIFY_DURATION = 1.35
startup_scan_active = True
startup_t0 = time.perf_counter()

def classify_cvzone_gesture(fingers):
    """Map cvzone fingersUp list to fist / palm / peace, or None."""
    if not fingers or len(fingers) < 5:
        return None
    count = sum(fingers[1:])
    if count == 0:
        return 'fist'
    if count == 4 and fingers[0] == 1:
        return 'palm'
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return 'peace'
    return None

def detect_hand_gesture(frame):
    """Returns current gesture label from cvzone, or None."""
    hands, _ = hand_detector.findHands(frame, draw=True)
    if not hands:
        return None
    return classify_cvzone_gesture(hand_detector.fingersUp(hands[0]))

def update_gesture_controls(detected_gesture):
    """Debounced fist/palm pause-resume and peace gimbal freeze."""
    global is_paused, gimbal_frozen, gimbal_freeze_capture
    global gesture_buffer, last_triggered_gesture

    gesture_buffer.append(detected_gesture)
    if len(gesture_buffer) > GESTURE_HOLD:
        gesture_buffer.pop(0)

    if len(gesture_buffer) == GESTURE_HOLD and len(set(gesture_buffer)) == 1:
        stable = gesture_buffer[0]
        if stable and stable != last_triggered_gesture:
            if stable == 'fist' and not is_paused:
                is_paused = True
                gimbal_frozen = False
                print("Fist -> PAUSED")
            elif stable == 'palm' and is_paused:
                is_paused = False
                print("Palm -> PLAYING")
            elif stable == 'peace' and not is_paused:
                gimbal_frozen = not gimbal_frozen
                gimbal_freeze_capture = gimbal_frozen
                if gimbal_frozen:
                    print("GIMBAL FROZEN")
                else:
                    print("GIMBAL UNFROZEN")
            last_triggered_gesture = stable
    elif detected_gesture is None:
        last_triggered_gesture = None

def draw_gesture_hint(frame, detected_gesture):
    """Bottom-left gesture label (local-style feedback on remote HUD)."""
    if not detected_gesture:
        return
    h = frame.shape[0]
    labels = {
        'fist': ("Gesture: FIST", (0, 0, 255)),
        'palm': ("Gesture: PALM", (0, 255, 0)),
        'peace': ("Gesture: PEACE", HUD_WARN),
    }
    text, color = labels[detected_gesture]
    cv2.putText(
        frame, text, (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
    )

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

def get_startup_phase(elapsed):
    if elapsed < STARTUP_SCAN_DURATION:
        return "scan", elapsed / STARTUP_SCAN_DURATION
    if elapsed < STARTUP_SCAN_DURATION + STARTUP_IDENTIFY_DURATION:
        identify_elapsed = elapsed - STARTUP_SCAN_DURATION
        return "identify", identify_elapsed / STARTUP_IDENTIFY_DURATION
    return "done", 1.0

def collect_cosmetic_targets(results, frame_w, frame_h):
    """Fake scan hits from current detections (visual only)."""
    targets = []
    if results[0].boxes is None or results[0].boxes.id is None:
        return targets

    boxes = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
    keypoints_xy = None
    if results[0].keypoints is not None and results[0].keypoints.xy is not None:
        keypoints_xy = results[0].keypoints.xy.cpu().numpy()

    for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
        x, y, w, h = box
        tx, ty = int(x), int(y)
        if keypoints_xy is not None and idx < len(keypoints_xy):
            nose = get_nose_point(keypoints_xy[idx])
            if nose is not None:
                tx, ty = nose
            else:
                torso = get_torso_center_point(keypoints_xy[idx])
                if torso is not None:
                    tx, ty = torso
        targets.append({"id": int(track_id), "x": tx, "y": ty})
    return targets

def draw_startup_laser_scan(frame, scan_progress, targets):
    h, w = frame.shape[:2]
    scan_y = int(np.clip(scan_progress, 0.0, 1.0) * h)
    laser_core = (255, 255, 120)
    laser_glow = HUD_ACCENT

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, scan_y), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)

    for offset, thickness, color in ((0, 2, laser_core), (-4, 1, laser_glow), (4, 1, laser_glow)):
        y = np.clip(scan_y + offset, 0, h - 1)
        cv2.line(frame, (0, y), (w, y), color, thickness, cv2.LINE_AA)

    for trail in range(1, 10):
        ty = scan_y - trail * 7
        if ty < 0:
            break
        fade = max(40, 180 - trail * 16)
        cv2.line(frame, (0, ty), (w, ty), (fade, fade, 0), 1, cv2.LINE_AA)

    pinged = 0
    for target in targets:
        if abs(target["y"] - scan_y) <= 22:
            pinged += 1
            cv2.circle(frame, (target["x"], target["y"]), 14, laser_glow, 2, cv2.LINE_AA)
            cv2.circle(frame, (target["x"], target["y"]), 4, laser_core, -1, cv2.LINE_AA)
            cv2.putText(
                frame, f"SIG {target['id']:02d}",
                (target["x"] + 16, target["y"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, laser_glow, 1, cv2.LINE_AA,
            )

    status = "LASER SCAN: SEARCHING FOR SUBJECTS..."
    cv2.putText(frame, status, (24, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, laser_glow, 2, cv2.LINE_AA)
    cv2.putText(
        frame, f"Sweep {int(scan_progress * 100):3d}%  |  Hits {pinged}",
        (24, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, HUD_DIM, 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "DRIIFT BOOT SEQUENCE",
        (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, HUD_ACCENT, 2, cv2.LINE_AA,
    )

def draw_startup_identify(frame, identify_progress, targets):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    visible_count = 0
    for i, target in enumerate(targets):
        reveal = np.clip((identify_progress * (len(targets) + 1)) - i, 0.0, 1.0)
        if reveal <= 0.0:
            continue
        visible_count += 1
        tx, ty = target["x"], target["y"]
        box_half = int(24 + 8 * reveal)
        alpha_color = (
            int(HUD_ACCENT[0] * reveal),
            int(HUD_ACCENT[1] * reveal),
            int(HUD_ACCENT[2] * reveal),
        )
        x1, y1 = tx - box_half, ty - box_half
        x2, y2 = tx + box_half, ty + box_half
        cv2.rectangle(frame, (x1, y1), (x2, y2), alpha_color, 2, cv2.LINE_AA)
        draw_lock_on_reticle(frame, tx, ty, locked=reveal > 0.65)
        cv2.putText(
            frame, f"ID {target['id']:02d} IDENTIFIED",
            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, alpha_color, 1, cv2.LINE_AA,
        )

    cv2.putText(
        frame, "SUBJECT IDENTIFICATION",
        (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, HUD_ACCENT, 2, cv2.LINE_AA,
    )
    label = "MAPPING TARGETS..." if identify_progress < 0.85 else "SUBJECTS IDENTIFIED"
    cv2.putText(frame, label, (24, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, HUD_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(
        frame, f"Confirmed {visible_count}/{max(len(targets), 1)}",
        (24, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, HUD_DIM, 1, cv2.LINE_AA,
    )

def draw_startup_sequence(frame, phase, phase_progress, targets):
    draw_cinematic_crosshair(frame)
    if phase == "scan":
        draw_startup_laser_scan(frame, phase_progress, targets)
    elif phase == "identify":
        draw_startup_identify(frame, phase_progress, targets)

def get_tracking_mode_label(is_paused, gimbal_frozen, has_subject_lock):
    if is_paused:
        return "STANDBY"
    if gimbal_frozen:
        return "GIMBAL HOLD"
    if has_subject_lock:
        return "LOCKED"
    return "SEARCHING"

def update_fps_counter():
    global fps_value, fps_frame_count, fps_last_tick
    fps_frame_count += 1
    now = time.perf_counter()
    elapsed = now - fps_last_tick
    if elapsed >= 0.5:
        fps_value = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_last_tick = now

def update_fake_battery():
    global fake_battery_pct, battery_last_tick
    now = time.perf_counter()
    elapsed = now - battery_last_tick
    if elapsed < 1.0:
        return
    battery_last_tick = now
    fake_battery_pct -= 0.04 * elapsed
    if fake_battery_pct < 18.0:
        fake_battery_pct = 88.0

def draw_cinematic_crosshair(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = (210, 210, 210)
    gap = 14
    arm = min(w, h) // 5
    thickness = 1

    cv2.line(frame, (cx - arm, cy), (cx - gap, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy), (cx + arm, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm), (cx, cy - gap), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap), (cx, cy + arm), color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, color, 1, cv2.LINE_AA)

    inset = 28
    corner = 34
    for ox, oy, dx, dy in (
        (inset, inset, 1, 1), (w - inset, inset, -1, 1),
        (inset, h - inset, 1, -1), (w - inset, h - inset, -1, -1),
    ):
        cv2.line(frame, (ox, oy), (ox + dx * corner, oy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (ox, oy), (ox, oy + dy * corner), color, 1, cv2.LINE_AA)

def draw_lock_on_reticle(frame, cx, cy, locked=True):
    pulse = 0.85 + 0.15 * abs(np.sin(time.perf_counter() * 4.5))
    size = int(34 * pulse)
    color = HUD_ACCENT if locked else HUD_DIM
    thickness = 2 if locked else 1
    half = size // 2
    gap = 7

    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    seg = half - gap
    cv2.line(frame, (x1, y1), (x1 + seg, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + seg), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - seg, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + seg), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + seg, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - seg), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - seg, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - seg), color, thickness, cv2.LINE_AA)

    cv2.line(frame, (cx - 10, cy), (cx - 3, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + 3, cy), (cx + 10, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 10), (cx, cy - 3), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 3), (cx, cy + 10), color, 1, cv2.LINE_AA)

def draw_battery_indicator(frame, battery_pct):
    h, w = frame.shape[:2]
    x, y = w - 150, 24
    pct = int(np.clip(battery_pct, 0, 100))
    label = f"PWR {pct:3d}%"
    color = HUD_ACCENT if pct > 35 else HUD_WARN if pct > 20 else HUD_ALERT
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = x, y + 10, 110, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), HUD_DIM, 1, cv2.LINE_AA)
    fill_w = int(bar_w * (pct / 100.0))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x + 1, bar_y + 1), (bar_x + fill_w - 1, bar_y + bar_h - 1), color, -1, cv2.LINE_AA)

def draw_hud_panel(frame, tracking_mode, target_confidence, fps, gimbal_xy, gimbal_frozen):
    h, w = frame.shape[:2]
    panel_h = 118
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

    mode_color = HUD_ACCENT
    if tracking_mode == "STANDBY":
        mode_color = HUD_ALERT
    elif tracking_mode == "GIMBAL HOLD":
        mode_color = HUD_WARN
    elif tracking_mode == "SEARCHING":
        mode_color = HUD_DIM

    cv2.putText(frame, "DRIIFT TRACKING HUD", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, HUD_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"MODE: {tracking_mode}", (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, mode_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"TARGET CONF: {target_confidence:5.1f}%", (16, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, HUD_ACCENT, 1, cv2.LINE_AA)

    move_x, move_y = gimbal_xy
    gimbal_label = "GIMBAL HOLD" if gimbal_frozen else "GIMBAL LIVE"
    gimbal_color = HUD_WARN if gimbal_frozen else (255, 255, 0)
    cv2.putText(frame, f"{gimbal_label}  x:{move_x:+.2f}  y:{move_y:+.2f}", (290, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, gimbal_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS {fps:4.1f}", (w - 96, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, HUD_DIM, 1, cv2.LINE_AA)

def draw_gimbal_pause_minimap(frame, frame_w, frame_h, subject_point, gimbal_xy):
    """
    Corner tactical map shown only while gimbal is frozen (peace-sign hold).
    Shows subject position, camera command direction, and tracking cone.
    """
    h, w = frame.shape[:2]
    map_w, map_h = 132, 108
    margin = 14
    x0, y0 = margin, h - map_h - margin
    pad = 10
    inner_x = x0 + pad
    inner_y = y0 + 22
    inner_w = map_w - 2 * pad
    inner_h = map_h - 32

    roi = frame[y0:y0 + map_h, x0:x0 + map_w]
    panel = roi.copy()
    cv2.rectangle(panel, (0, 0), (map_w - 1, map_h - 1), (18, 18, 18), -1)
    cv2.addWeighted(panel, 0.84, roi, 0.16, 0, roi)
    cv2.rectangle(frame, (x0, y0), (x0 + map_w - 1, y0 + map_h - 1), HUD_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(
        frame, "TAC MAP", (x0 + 6, y0 + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, HUD_WARN, 1, cv2.LINE_AA,
    )

    cv2.rectangle(frame, (inner_x, inner_y), (inner_x + inner_w, inner_y + inner_h), HUD_DIM, 1, cv2.LINE_AA)
    cx = inner_x + inner_w // 2
    cy = inner_y + inner_h // 2
    cv2.drawMarker(frame, (cx, cy), HUD_DIM, cv2.MARKER_CROSS, 6, 1, lineType=cv2.LINE_AA)

    if subject_point is not None:
        sx, sy = subject_point
        dot_x = inner_x + int(np.clip(sx / max(frame_w, 1), 0.0, 1.0) * inner_w)
        dot_y = inner_y + int(np.clip(sy / max(frame_h, 1), 0.0, 1.0) * inner_h)

        dx, dy = dot_x - cx, dot_y - cy
        angle = float(np.arctan2(dy, dx))
        cone_half = np.radians(22)
        radius = min(inner_w, inner_h) // 2 - 2
        cone_pts = [(cx, cy)]
        for a in np.linspace(angle - cone_half, angle + cone_half, 14):
            cone_pts.append((int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a))))
        cv2.fillPoly(frame, [np.array(cone_pts, dtype=np.int32)], (70, 130, 130))
        cv2.circle(frame, (dot_x, dot_y), 3, HUD_ACCENT, -1, lineType=cv2.LINE_AA)

    gx, gy = gimbal_xy
    mag = float(np.hypot(gx, gy))
    arrow_scale = min(inner_w, inner_h) // 2 - 3
    if mag >= 0.05:
        ax = int(cx + gx * arrow_scale)
        ay = int(cy + gy * arrow_scale)
        cv2.arrowedLine(
            frame, (cx, cy), (ax, ay), (140, 140, 255), 1,
            tipLength=0.35, line_type=cv2.LINE_AA,
        )

    legend_y = y0 + map_h - 8
    cv2.circle(frame, (x0 + 8, legend_y - 18), 2, HUD_ACCENT, -1, lineType=cv2.LINE_AA)
    cv2.putText(frame, "SUB", (x0 + 14, legend_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_DIM, 1, cv2.LINE_AA)
    cv2.line(frame, (x0 + 42, legend_y - 17), (x0 + 54, legend_y - 17), (140, 140, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "CAM", (x0 + 58, legend_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_DIM, 1, cv2.LINE_AA)
    cv2.line(frame, (x0 + 84, legend_y - 17), (x0 + 92, legend_y - 19), HUD_ACCENT, 1, cv2.LINE_AA)
    cv2.line(frame, (x0 + 84, legend_y - 17), (x0 + 92, legend_y - 15), HUD_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, "CONE", (x0 + 96, legend_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_DIM, 1, cv2.LINE_AA)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    if not check_arduino_connection():
        handle_connection_loss()
        if reconnect_arduino():
            connection_lost = False
            print("Arduino reconnected.")
    else:
        drain_arduino_inbox()

    if startup_scan_active:
        elapsed = time.perf_counter() - startup_t0
        phase, phase_progress = get_startup_phase(elapsed)
        cosmetic_targets = []
        boot_results = model.track(frame, persist=True, classes=0, conf=0.8, iou=0.5, verbose=False)
        cosmetic_targets = collect_cosmetic_targets(boot_results, frame_w, frame_h)
        draw_startup_sequence(frame, phase, phase_progress, cosmetic_targets)
        send_tracking_to_arduino(None, frame_w, frame_h, active=False)
        show_frame(frame)
        if phase == "done":
            startup_scan_active = False
            print("TRACKING ONLINE")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    detected_gesture = detect_hand_gesture(frame)
    update_gesture_controls(detected_gesture)

    update_fps_counter()
    update_fake_battery()

    coordinate_lines = []
    gimbal_output = (0.0, 0.0)
    target_confidence = 0.0
    subject_lock_point = None

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
            if subject_track_id is not None and subject_track_id in candidate_scores:
                target_confidence = candidate_scores[subject_track_id][0] * 100.0

            box_confs = None
            if results[0].boxes.conf is not None:
                box_confs = results[0].boxes.conf.cpu().numpy()

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
                    subject_lock_point = (display_x, display_y)
                    gimbal_output = compute_gimbal_output((display_x, display_y), frame_w, frame_h)
                    if box_confs is not None and idx < len(box_confs):
                        det_conf = float(box_confs[idx]) * 100.0
                        target_confidence = 0.65 * target_confidence + 0.35 * det_conf
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

    if gimbal_frozen and not is_paused:
        if gimbal_freeze_capture:
            frozen_gimbal_output = gimbal_output
            gimbal_freeze_capture = False
        gimbal_output = frozen_gimbal_output

    move_x, move_y = gimbal_output
    tracking_mode = get_tracking_mode_label(is_paused, gimbal_frozen, subject_lock_point is not None)
    draw_gesture_hint(frame, detected_gesture)
    draw_cinematic_crosshair(frame)
    if subject_lock_point is not None and not is_paused:
        draw_lock_on_reticle(frame, subject_lock_point[0], subject_lock_point[1], locked=True)
    draw_battery_indicator(frame, fake_battery_pct)
    draw_hud_panel(
        frame,
        tracking_mode=tracking_mode,
        target_confidence=target_confidence,
        fps=fps_value,
        gimbal_xy=(move_x, move_y),
        gimbal_frozen=gimbal_frozen and not is_paused,
    )
    if gimbal_frozen and not is_paused:
        draw_gimbal_pause_minimap(
            frame, frame_w, frame_h, subject_lock_point, frozen_gimbal_output,
        )

    freeze_tag = " FROZEN" if gimbal_frozen and not is_paused else ""
    if is_paused:
        print("TRACKING PAUSED | GIMBAL x:+0.00 y:+0.00")
    elif coordinate_lines:
        print(" | ".join(coordinate_lines) + f" | GIMBAL x:{move_x:+.2f} y:{move_y:+.2f}{freeze_tag}")
    else:
        print(f"No tracked coordinates | GIMBAL x:{move_x:+.2f} y:{move_y:+.2f}{freeze_tag}")

    tracking_active = (
        not is_paused
        and not gimbal_frozen
        and subject_lock_point is not None
    )
    send_tracking_to_arduino(subject_lock_point, frame_w, frame_h, tracking_active)

    show_frame(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        is_paused = not is_paused
        if is_paused:
            gimbal_frozen = False
        print("PAUSED" if is_paused else "RESUME")
    if key == ord('r'):
        send_arduino_command("RESET")
        print("Sent RESET — servos re-centering.")

if arduino is not None:
    try:
        arduino.close()
    except Exception:
        pass

cap.release()
cv2.destroyAllWindows()