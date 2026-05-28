import cv2
import mediapipe as mp
import threading
import time
import math
import json
import os
import glob
from datetime import datetime
import numpy as np
import serial
import serial.tools.list_ports
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# -- MediaPipe Hands setup -----------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# -- Webcam --------------------------------------------------------------------
cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARN] Webcam not found - running in DEMO mode")
        cap = None
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[OK] Webcam opened")
except Exception as e:
    print(f"[WARN] Webcam error: {e} - running in DEMO mode")
    cap = None

# -- Arduino UNO Serial Connection ---------------------------------------------
SERIAL_PORT = "COM7"
SERIAL_BAUD = 9600
SERIAL_TIMEOUT = 1

ser = None
serial_status = "DISCONNECTED"
serial_last_response = ""

def connect_serial():
    global ser, serial_status
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=SERIAL_TIMEOUT)
        time.sleep(2)
        while ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"[Arduino] {line}")
        serial_status = f"CONNECTED ({SERIAL_PORT})"
        print(f"[OK] Arduino UNO connected on {SERIAL_PORT} at {SERIAL_BAUD} baud")
        return True
    except Exception as e:
        serial_status = f"ERROR: {e}"
        print(f"[WARN] Serial connect failed: {e}")
        ser = None
        return False

def send_to_arduino(channel: str, angle: int):
    global ser, serial_status, serial_last_response
    if ser is None:
        return False
    try:
        cmd = f"{channel},{angle}\n"
        ser.write(cmd.encode('utf-8'))
        ser.flush()
        if ser.in_waiting:
            resp = ser.readline().decode('utf-8', errors='ignore').strip()
            serial_last_response = resp
        return True
    except Exception as e:
        serial_status = f"LOST: {e}"
        print(f"[WARN] Serial send failed: {e}")
        ser = None
        return False

connect_serial()

# -- Shared state (6-DOF) -----------------------------------------------------
state = {
    "base":         90,
    "shoulder":     170,
    "elbow":        10,
    "wrist_pitch":  0,
    "wrist_roll":   90,
    "gripper":      50,
    "hand_x":       0.5,
    "hand_y":       0.5,
    "detected":     False,
    "direction":    "CENTER",
}
_lock = threading.Lock()

DEADBAND = 2
_last_sent = {"B": 90, "S": 170, "E": 10, "W": 0, "R": 90, "G": 50}
_last_send_time = {"B": 0, "S": 0, "E": 0, "W": 0, "R": 0, "G": 0}
_min_send_interval = 0.03

# -- EMA Smoothing (reduces jitter on all gesture inputs) ----------------------
_EMA_ALPHA = 0.25  # 0.0 = frozen, 1.0 = no smoothing. 0.25 = smooth but responsive
_ema = {"B": 90.0, "S": 170.0, "E": 10.0, "W": 0.0, "R": 90.0, "G": 50.0}

def smooth(channel: str, raw: int) -> int:
    """Apply Exponential Moving Average to reduce jitter."""
    _ema[channel] += _EMA_ALPHA * (raw - _ema[channel])
    return int(round(_ema[channel]))

def update_joint_and_send(channel: str, angle: int):
    global _last_sent, _last_send_time
    now = time.time()
    if abs(angle - _last_sent[channel]) >= DEADBAND and (now - _last_send_time[channel]) >= _min_send_interval:
        send_to_arduino(channel, angle)
        _last_sent[channel] = angle
        _last_send_time[channel] = now

# -- Task Recording & Playback -------------------------------------------------
TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tasks')
os.makedirs(TASKS_DIR, exist_ok=True)

recording = False
recording_task_name = ""  # name set BEFORE recording begins
recording_buffer = []
recording_start_time = 0
_last_record_time = 0
RECORD_INTERVAL = 0.1  # capture keyframe every 100ms

playing = False
play_task_name = ""
_play_thread = None
_play_stop_event = threading.Event()

def capture_keyframe():
    """Append current joint angles as a keyframe if recording."""
    global _last_record_time
    if not recording:
        return
    now = time.time()
    if now - _last_record_time < RECORD_INTERVAL:
        return
    _last_record_time = now
    with _lock:
        kf = {
            "t": round(now - recording_start_time, 3),
            "B": state["base"],
            "S": state["shoulder"],
            "E": state["elbow"],
            "W": state["wrist_pitch"],
            "R": state["wrist_roll"],
            "G": state["gripper"]
        }
    recording_buffer.append(kf)

def playback_worker(task_data):
    """Background thread that replays keyframes."""
    global playing, play_task_name
    keyframes = task_data.get("keyframes", [])
    if not keyframes:
        playing = False
        play_task_name = ""
        return

    start = time.time()
    for i, kf in enumerate(keyframes):
        if _play_stop_event.is_set():
            break
        # Wait until the right time
        target_time = start + kf["t"]
        wait = target_time - time.time()
        if wait > 0:
            _play_stop_event.wait(timeout=wait)
            if _play_stop_event.is_set():
                break

        # Apply joint angles
        with _lock:
            state["base"]        = kf["B"]
            state["shoulder"]    = kf["S"]
            state["elbow"]       = kf["E"]
            state["wrist_pitch"] = kf.get("W", 90)
            state["wrist_roll"]  = kf.get("R", 90)
            state["gripper"]     = kf["G"]
            state["detected"]    = False

        update_joint_and_send("B", kf["B"])
        update_joint_and_send("S", kf["S"])
        update_joint_and_send("E", kf["E"])
        update_joint_and_send("W", kf.get("W", 90))
        update_joint_and_send("R", kf.get("R", 90))
        update_joint_and_send("G", kf["G"])

    playing = False
    play_task_name = ""

# -- Gesture Helpers (6-DOF) ---------------------------------------------------

def _vector_angle(v1, v2):
    """Angle in degrees between two 2D vectors."""
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def wrist_x_to_base(wx: float) -> int:
    """DOF 1: Wrist X 0..1 → Base 10°..170°"""
    return max(10, min(170, int(10 + wx * 160)))


def wrist_y_to_shoulder(wy: float) -> int:
    """DOF 2: Wrist Y → Shoulder 170°(up)..0°(down).
    Uses compressed input range 0.10..0.95 so the hand doesn't need to
    reach the very edge of the camera frame to hit 170°."""
    y_min = 0.10   # hand at ~top of usable frame → 170°
    y_max = 0.95   # hand at bottom of frame       →   0°
    ratio = (wy - y_min) / (y_max - y_min)
    ratio = max(0.0, min(1.0, ratio))
    return max(0, min(170, int(170 - ratio * 170)))


def finger_curl_to_elbow(landmarks) -> int:
    """
    DOF 3: Index finger curl → Elbow angle.
    Open palm / straight index → 10°  (elbow extended).
    Fully curled index finger  → 170° (elbow bent down).
    Uses tip-to-MCP distance normalized by hand size.
    Range: 10° – 170°.
    """
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    hand_size = math.sqrt((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)
    if hand_size < 0.001:
        return 10

    # Index finger extension: tip(8) to MCP(5) distance, normalized
    tip = landmarks[8]
    mcp = landmarks[5]
    d = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
    extension = d / hand_size

    # Calibration (tuned to real hand proportions):
    # extended ≈ 0.85 (straight finger), fully curled ≈ 0.25
    # curl 0.0 = fully extended → 10°, curl 1.0 = fully curled → 170°
    curl = (0.85 - extension) / 0.60
    curl = max(0.0, min(1.0, curl))
    return max(10, min(170, int(10 + curl * 160)))


def fist_openness_to_wrist_pitch(landmarks) -> int:
    """
    DOF 4: Wrist pitch — average curl of middle, ring, pinky fingers ONLY.
    (Index finger is excluded so it doesn't interfere with elbow control;
     thumb is excluded so it doesn't interfere with wrist roll.)
    Open hand (fingers straight up) → 0° (min).
    All three fingers curled        → 170° (max).
    Range: 0° – 170°.
    """
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    hand_size = math.sqrt((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)
    if hand_size < 0.001:
        return 0

    # Measure tip-to-MCP distance for middle, ring, pinky only
    tips = [12, 16, 20]   # middle, ring, pinky fingertips
    mcps = [ 9, 13, 17]   # corresponding MCP joints

    ratios = []
    for t, m in zip(tips, mcps):
        tip = landmarks[t]
        mcp = landmarks[m]
        d = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
        ratios.append(d / hand_size)

    avg = sum(ratios) / len(ratios)

    # Calibration thresholds (MediaPipe tip-to-MCP ratios):
    # extended ≈ 0.85 (fingers straight), fully curled ≈ 0.25
    # curl 0 = open → 0°, curl 1 = fully curled → 170°
    curl = (0.85 - avg) / 0.60
    curl = max(0.0, min(1.0, curl))
    return max(0, min(170, int(curl * 170)))


def thumb_spread_to_wrist_roll(landmarks) -> int:
    """
    DOF 5: Thumb spread angle.
    Measures angle between:
      - Vector: thumb base(2) → thumb tip(4)
      - Vector: index knuckle(5) → index tip(8)
    Wide spread (~90°) → wrist roll at 160°.
    Thumb close (~0°) → wrist roll at 20°.
    """
    # Thumb vector: landmark 2 → landmark 4
    tb = landmarks[2]
    tt = landmarks[4]
    v_thumb = (tt.x - tb.x, tt.y - tb.y)

    # Index vector: landmark 5 → landmark 8
    ik = landmarks[5]
    it = landmarks[8]
    v_index = (it.x - ik.x, it.y - ik.y)

    spread_angle = _vector_angle(v_thumb, v_index)  # 0° = parallel, ~90° = spread

    # Map: 0° spread → 20°, 90° spread → 160°
    ratio = max(0.0, min(1.0, spread_angle / 90.0))
    return max(20, min(160, int(20 + ratio * 140)))


def pinky_ring_gap_to_gripper(landmarks) -> int:
    """
    DOF 6: Gap between ring finger tip (16) and pinky finger tip (20).
    Large gap (fingers spread) → gripper open  (90°).
    Small gap (fingers together) → gripper closed (0°).
    Uses 2D distance for higher precision at small gaps.
    """
    ring_tip  = landmarks[16]
    pinky_tip = landmarks[20]

    # 2D distance (x, y only; z-depth is excluded for precision)
    gap = math.sqrt((ring_tip.x - pinky_tip.x)**2 + (ring_tip.y - pinky_tip.y)**2)

    # Calibration: fingers touching ≈ 0.005, spread ≈ 0.10
    min_gap = 0.005
    max_gap = 0.10
    ratio = max(0.0, min(1.0, (gap - min_gap) / (max_gap - min_gap)))
    return max(0, min(90, int(ratio * 90)))


def dir_label(angle: int) -> str:
    if angle < 60:  return "LEFT"
    if angle > 120: return "RIGHT"
    return "CENTER"

# -- Frame generator -----------------------------------------------------------
def gen_frames():
    frame_count = 0

    while True:
        # -- DEMO MODE ---------------------------------------------------------
        if cap is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            h, w = 480, 640
            frame_count += 1

            demo_base = max(10, min(170, int(90 + 80 * np.sin(frame_count * 0.025))))
            demo_sh   = max(10, min(170, int(90 + 80 * np.sin(frame_count * 0.018))))
            demo_el   = max(10, min(170, int(90 + 80 * np.sin(frame_count * 0.03))))
            demo_wp   = max(10, min(170, int(90 + 80 * np.sin(frame_count * 0.022))))
            demo_wr   = max(20, min(160, int(90 + 70 * np.sin(frame_count * 0.027))))
            demo_gr   = max(10, min(90,  int(50 + 40 * np.sin(frame_count * 0.02))))

            with _lock:
                state["base"]        = demo_base
                state["shoulder"]    = demo_sh
                state["elbow"]       = demo_el
                state["wrist_pitch"] = demo_wp
                state["wrist_roll"]  = demo_wr
                state["gripper"]     = demo_gr
                state["detected"]    = True
                state["direction"]   = dir_label(demo_base)

            update_joint_and_send("B", demo_base)
            update_joint_and_send("S", demo_sh)
            update_joint_and_send("E", demo_el)
            update_joint_and_send("W", demo_wp)
            update_joint_and_send("R", demo_wr)
            update_joint_and_send("G", demo_gr)

            capture_keyframe()

            # Draw dark background
            for y in range(h):
                v = int(15 + y * 25 / h)
                frame[y, :] = (v, v+5, v+20)

            cv2.putText(frame, "DEMO MODE  (connect webcam for real tracking)",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 130, 255), 1)
            cv2.putText(frame, f"B:{demo_base} S:{demo_sh} E:{demo_el} W:{demo_wp} R:{demo_wr} G:{demo_gr}",
                        (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 160), 2)

            # 6 progress bars
            colors = [(0,255,160), (255,180,50), (180,100,255), (255,217,61), (255,107,157), (100,220,255)]
            vals   = [demo_base, demo_sh, demo_el, demo_wr, demo_wp, demo_gr]
            maxes  = [170, 170, 170, 160, 170, 90]
            for i, (c, v, mx) in enumerate(zip(colors, vals, maxes)):
                y0 = i * 6
                bw = int(v / mx * w)
                cv2.rectangle(frame, (0, y0), (w, y0+4), (30,30,30), -1)
                cv2.rectangle(frame, (0, y0), (bw, y0+4), c, -1)

            time.sleep(0.033)

        # -- REAL WEBCAM MODE --------------------------------------------------
        else:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands_detector.process(rgb)

            base_angle = state["base"]
            sh_angle   = state["shoulder"]
            el_angle   = state["elbow"]
            wp_angle   = state["wrist_pitch"]
            wr_angle   = state["wrist_roll"]
            gr_angle   = state["gripper"]

            if result.multi_hand_landmarks and not playing:
                lm = result.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )

                wrist_x = lm.landmark[0].x
                wrist_y = lm.landmark[0].y

                new_base = smooth("B", wrist_x_to_base(wrist_x))
                new_sh   = smooth("S", wrist_y_to_shoulder(wrist_y))
                new_el   = smooth("E", finger_curl_to_elbow(lm.landmark))
                new_wp   = smooth("W", fist_openness_to_wrist_pitch(lm.landmark))
                new_wr   = smooth("R", thumb_spread_to_wrist_roll(lm.landmark))
                new_gr   = smooth("G", pinky_ring_gap_to_gripper(lm.landmark))

                if abs(new_base - base_angle) > DEADBAND: base_angle = new_base
                if abs(new_sh - sh_angle)     > DEADBAND: sh_angle   = new_sh
                if abs(new_el - el_angle)     > DEADBAND: el_angle   = new_el
                if abs(new_wp - wp_angle)     > DEADBAND: wp_angle   = new_wp
                if abs(new_wr - wr_angle)     > DEADBAND: wr_angle   = new_wr
                if abs(new_gr - gr_angle)     > DEADBAND: gr_angle   = new_gr

                with _lock:
                    state["base"]        = base_angle
                    state["shoulder"]    = sh_angle
                    state["elbow"]       = el_angle
                    state["wrist_pitch"] = wp_angle
                    state["wrist_roll"]  = wr_angle
                    state["gripper"]     = gr_angle
                    state["hand_x"]      = wrist_x
                    state["hand_y"]      = wrist_y
                    state["detected"]    = True
                    state["direction"]   = dir_label(base_angle)

                update_joint_and_send("B", base_angle)
                update_joint_and_send("S", sh_angle)
                update_joint_and_send("E", el_angle)
                update_joint_and_send("W", wp_angle)
                update_joint_and_send("R", wr_angle)
                update_joint_and_send("G", gr_angle)

                capture_keyframe()

            else:
                with _lock:
                    state["detected"]  = False
                    state["direction"] = dir_label(state["base"])

            # -- Overlay -------------------------------------------------------
            # 6 colour-coded progress bars at top
            colors = [(0,255,160), (255,180,50), (180,100,255), (255,217,61), (255,107,157), (100,220,255)]
            vals   = [base_angle, sh_angle, el_angle, wr_angle, wp_angle, gr_angle]
            maxes  = [170, 170, 170, 160, 170, 90]
            for i, (c, v, mx) in enumerate(zip(colors, vals, maxes)):
                y0 = i * 6
                bw = int(v / mx * w)
                cv2.rectangle(frame, (0, y0), (w, y0+4), (25,25,25), -1)
                cv2.rectangle(frame, (0, y0), (bw, y0+4), c, -1)

            # Bar labels (compact)
            labels = ["BASE", "SHOULDER", "ELBOW", "WR.ROLL", "WR.PITCH", "GRIPPER"]
            colors_txt = [(0,255,160), (255,180,50), (180,100,255), (255,217,61), (255,107,157), (100,220,255)]
            for i, (lbl, c) in enumerate(zip(labels, colors_txt)):
                y0 = i * 6
                cv2.putText(frame, lbl, (4, y0+4), cv2.FONT_HERSHEY_SIMPLEX, 0.2, c, 1)

            # Serial status pill
            ser_connected = ser is not None
            ser_clr = (0, 180, 60) if ser_connected else (40, 40, 60)
            ser_txt = f"Arduino: {SERIAL_PORT}" if ser_connected else "Arduino: OFFLINE"
            cv2.rectangle(frame, (w-200, h-40), (w-10, h-12), ser_clr, -1)
            cv2.rectangle(frame, (w-200, h-40), (w-10, h-12), (255,255,255), 1)
            cv2.putText(frame, ser_txt, (w-194, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Hand detection pill
            detected = state["detected"]
            pill_clr = (0, 200, 80) if detected else (60, 60, 80)
            pill_txt = "HAND DETECTED" if detected else "NO HAND"
            cv2.rectangle(frame, (w-190, 36), (w-10, 62), pill_clr, -1)
            cv2.rectangle(frame, (w-190, 36), (w-10, 62), (255,255,255), 1)
            cv2.putText(frame, pill_txt, (w-184, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

# -- Flask routes --------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with _lock:
        s = dict(state)
    s["serial_status"]    = serial_status
    s["serial_port"]      = SERIAL_PORT
    s["serial_connected"] = ser is not None
    s["serial_last_resp"] = serial_last_response
    s["recording"]           = recording
    s["recording_task_name"] = recording_task_name
    s["recording_frames"]    = len(recording_buffer)
    s["recording_elapsed"]   = round(time.time() - recording_start_time, 1) if recording else 0
    s["playing"]             = playing
    s["play_task_name"]      = play_task_name
    return jsonify(s)

@app.route('/manual', methods=['POST'])
def manual():
    data    = request.json or {}
    channel = data.get('channel', 'B')
    angle   = max(0, min(180, int(data.get('angle', 90))))

    with _lock:
        if   channel == 'B': state["base"]        = angle
        elif channel == 'S': state["shoulder"]     = angle
        elif channel == 'E': state["elbow"]        = angle
        elif channel == 'W': state["wrist_pitch"]  = angle
        elif channel == 'R': state["wrist_roll"]   = angle
        elif channel == 'G': state["gripper"]      = angle
        state["detected"] = False
        if channel == 'B': state["direction"] = dir_label(angle)

    update_joint_and_send(channel, angle)
    capture_keyframe()
    return jsonify({"ok": True, "channel": channel, "angle": angle})

# -- Task Recording & Playback Routes -----------------------------------------
@app.route('/task/record/start', methods=['POST'])
def task_record_start():
    global recording, recording_task_name, recording_buffer, recording_start_time, _last_record_time
    if playing:
        return jsonify({"ok": False, "error": "Cannot record while playing"})
    if recording:
        return jsonify({"ok": False, "error": "Already recording"})

    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({"ok": False, "error": "Task name is required"})

    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_name:
        return jsonify({"ok": False, "error": "Invalid task name"})

    recording_task_name = safe_name
    recording_buffer = []
    recording_start_time = time.time()
    _last_record_time = 0
    recording = True

    # Capture initial position as first keyframe
    with _lock:
        kf = {
            "t": 0.0,
            "B": state["base"],
            "S": state["shoulder"],
            "E": state["elbow"],
            "W": state["wrist_pitch"],
            "R": state["wrist_roll"],
            "G": state["gripper"]
        }
    recording_buffer.append(kf)
    print(f"[TASK] Recording started → '{safe_name}'")
    return jsonify({"ok": True, "name": safe_name})

@app.route('/task/record/stop', methods=['POST'])
def task_record_stop():
    global recording, recording_task_name
    if not recording:
        return jsonify({"ok": False, "error": "Not recording"})

    recording = False
    name = recording_task_name

    if not name:
        return jsonify({"ok": False, "error": "No task name set"})

    task_data = {
        "name": name,
        "created": datetime.now().isoformat(),
        "keyframes": list(recording_buffer)
    }
    filepath = os.path.join(TASKS_DIR, f"{name}.json")
    with open(filepath, 'w') as f:
        json.dump(task_data, f, indent=2)

    frame_count = len(recording_buffer)
    duration = round(recording_buffer[-1]["t"], 1) if recording_buffer else 0
    recording_task_name = ""
    print(f"[TASK] Saved '{name}' — {frame_count} keyframes, {duration}s")
    return jsonify({"ok": True, "name": name, "frames": frame_count, "duration": duration})

@app.route('/task/record/discard', methods=['POST'])
def task_record_discard():
    """Cancel and discard a recording without saving."""
    global recording, recording_task_name, recording_buffer
    was_recording = recording
    recording = False
    recording_task_name = ""
    recording_buffer = []
    print(f"[TASK] Recording discarded")
    return jsonify({"ok": True, "was_recording": was_recording})

@app.route('/task/list')
def task_list():
    tasks = []
    for fp in sorted(glob.glob(os.path.join(TASKS_DIR, '*.json'))):
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            kfs = data.get("keyframes", [])
            tasks.append({
                "name": data.get("name", os.path.basename(fp)),
                "created": data.get("created", ""),
                "keyframes": len(kfs),
                "duration": round(kfs[-1]["t"], 1) if kfs else 0
            })
        except Exception:
            pass
    return jsonify(tasks)

@app.route('/task/search')
def task_search():
    """Search tasks by partial name match."""
    q = request.args.get('q', '').strip().lower()
    tasks = []
    for fp in sorted(glob.glob(os.path.join(TASKS_DIR, '*.json'))):
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            name = data.get("name", "")
            if q in name.lower():
                kfs = data.get("keyframes", [])
                tasks.append({
                    "name": name,
                    "created": data.get("created", ""),
                    "keyframes": len(kfs),
                    "duration": round(kfs[-1]["t"], 1) if kfs else 0
                })
        except Exception:
            pass
    return jsonify(tasks)

@app.route('/task/play', methods=['POST'])
def task_play():
    global playing, play_task_name, _play_thread, _play_stop_event
    if recording:
        return jsonify({"ok": False, "error": "Cannot play while recording"})
    if playing:
        return jsonify({"ok": False, "error": "Already playing a task"})

    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({"ok": False, "error": "Task name is required"})

    filepath = os.path.join(TASKS_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        return jsonify({"ok": False, "error": f"Task '{name}' not found"})

    with open(filepath, 'r') as f:
        task_data = json.load(f)

    playing = True
    play_task_name = name
    _play_stop_event.clear()
    _play_thread = threading.Thread(target=playback_worker, args=(task_data,), daemon=True)
    _play_thread.start()
    kfs = task_data.get("keyframes", [])
    duration = round(kfs[-1]["t"], 1) if kfs else 0
    print(f"[TASK] Playing '{name}' — {len(kfs)} keyframes, {duration}s")
    return jsonify({"ok": True, "name": name, "keyframes": len(kfs), "duration": duration})

@app.route('/task/stop', methods=['POST'])
def task_stop():
    global playing, play_task_name
    _play_stop_event.set()
    playing = False
    play_task_name = ""
    print(f"[TASK] Playback stopped")
    return jsonify({"ok": True})

@app.route('/task/delete', methods=['POST'])
def task_delete():
    data = request.json or {}
    name = data.get('name', '').strip()
    filepath = os.path.join(TASKS_DIR, f"{name}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"[TASK] Deleted '{name}'")
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Task not found"})

# -- AI Smart Presets (ML-powered) --------------------------------------------
def _auto_name_preset(b, s, e, w, r, g):
    """Generate a descriptive name from joint angles."""
    parts = []
    # Base direction
    if b < 60:
        parts.append("Left")
    elif b > 120:
        parts.append("Right")
    else:
        parts.append("Center")
    # Shoulder height
    if s > 130:
        parts.append("Raised")
    elif s < 50:
        parts.append("Low")
    # Elbow bend
    if e < 50:
        parts.append("Extended")
    elif e > 130:
        parts.append("Folded")
    else:
        parts.append("Bent")
    # Gripper
    if g <= 25:
        parts.append("· Grip")
    elif g >= 70:
        parts.append("· Open")
    else:
        parts.append("· Half")
    return " ".join(parts)


@app.route('/smart-presets')
def smart_presets():
    """Cluster all recorded keyframes to discover common arm positions."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return jsonify({"ok": False, "error": "scikit-learn not installed. Run: pip install scikit-learn", "presets": []})

    # Collect all keyframes from all saved tasks
    all_keyframes = []
    task_files = glob.glob(os.path.join(TASKS_DIR, '*.json'))
    for fp in task_files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            for kf in data.get("keyframes", []):
                all_keyframes.append([
                    kf.get("B", 90), kf.get("S", 90), kf.get("E", 90),
                    kf.get("W", 90), kf.get("R", 90), kf.get("G", 50)
                ])
        except Exception:
            pass

    if len(all_keyframes) < 3:
        return jsonify({"ok": False, "error": "Not enough data \u2014 record more tasks first", "presets": []})

    # Determine cluster count from unique poses
    unique_poses = len(set(map(tuple, all_keyframes)))
    n_clusters = min(6, max(2, unique_poses // 5))
    n_clusters = min(n_clusters, len(all_keyframes))

    X = np.array(all_keyframes, dtype=float)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    presets = []
    for i, center in enumerate(kmeans.cluster_centers_):
        b, s, e, w, r, g = [int(round(v)) for v in center]
        count = int(np.sum(kmeans.labels_ == i))
        presets.append({
            "name": _auto_name_preset(b, s, e, w, r, g),
            "B": b, "S": s, "E": e, "W": w, "R": r, "G": g,
            "frequency": count,
            "pct": round(count / len(all_keyframes) * 100, 1)
        })

    presets.sort(key=lambda p: p["frequency"], reverse=True)
    print(f"[ML] Generated {len(presets)} smart presets from {len(all_keyframes)} keyframes")
    return jsonify({
        "ok": True,
        "presets": presets,
        "total_keyframes": len(all_keyframes),
        "task_count": len(task_files)
    })


@app.route('/reconnect', methods=['POST'])
def reconnect():
    ok = connect_serial()
    if ok:
        send_to_arduino("B", state["base"])
        time.sleep(0.05)
        send_to_arduino("S", state["shoulder"])
        time.sleep(0.05)
        send_to_arduino("E", state["elbow"])
        time.sleep(0.05)
        send_to_arduino("W", state["wrist_pitch"])
        time.sleep(0.05)
        send_to_arduino("R", state["wrist_roll"])
        time.sleep(0.05)
        send_to_arduino("G", state["gripper"])
    return jsonify({"ok": ok, "status": serial_status})

# -- Entry point ---------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 50)
    print("  Apprentice Arm — 6-DOF Gesture Controller")
    print(f"  Serial: {serial_status}")
    print("  Open -> http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)