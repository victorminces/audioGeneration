"""Hand-tracking child process: camera + MediaPipe in their own interpreter.

Runs completely isolated from the instrument's audio/UI process (own GIL, own
core). Sends small messages over queues:
  state_q:  ("state", x, y, gesture, fps)   ~15/s   gesture: "pinch"|"two"|None
            ("error", message)              on fatal problems
  frame_q:  ("frame", small_bgr_ndarray)    ~10/s   for the dimmed preview
"""
import os
import time

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "hand_landmarker.task")
DETECT_FPS  = 15
PREVIEW_FPS = 10
PREVIEW_W   = 212


def _classify(lm, prev):
    """21 landmarks + previous gesture -> (gesture, x, y) in normalized camera
    coords. Hysteresis: once a gesture is active its thresholds relax, so
    landmark jitter doesn't flicker it off mid-take."""
    def p(i):
        return np.array([lm[i].x, lm[i].y])
    wrist = p(0)

    def extended(tip, pip, slack):
        return np.linalg.norm(p(tip) - wrist) > slack * np.linalg.norm(p(pip) - wrist)

    pinch_d = np.linalg.norm(p(4) - p(8))
    hand_scale = np.linalg.norm(p(9) - wrist) + 1e-6      # wrist -> middle base
    if pinch_d / hand_scale < (0.50 if prev == "pinch" else 0.35):
        pt = (p(4) + p(8)) / 2                             # pinch point
        return "pinch", pt[0], pt[1]
    if prev == "two":                                      # sticky: only index+middle matter
        two = extended(8, 6, 1.02) and extended(12, 10, 1.02)
    else:
        two = (extended(8, 6, 1.12) and extended(12, 10, 1.12)
               and not extended(16, 14, 1.12) and not extended(20, 18, 1.12))
    if two:
        pt = p(8)                                          # index fingertip
        return "two", pt[0], pt[1]
    return None, p(8)[0], p(8)[1]


def run(state_q, frame_q, stop_ev):
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            state_q.put(("error", "no camera found"))
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1, running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.6, min_tracking_confidence=0.5)
        hands = vision.HandLandmarker.create_from_options(opts)

        last_detect = 0.0
        last_frame = 0.0
        last_ts_ms = 0
        fps = 0.0
        x = y = 0.5
        g = None
        last_seen = 0.0                         # last time a hand was detected
        while not stop_ev.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.perf_counter()

            if now - last_frame >= 1.0 / PREVIEW_FPS and frame_q.qsize() < 2:
                h = int(frame.shape[0] * PREVIEW_W / frame.shape[1])
                small = cv2.flip(cv2.resize(frame, (PREVIEW_W, h)), 1)
                frame_q.put(("frame", small))
                last_frame = now

            if now - last_detect < 1.0 / DETECT_FPS:
                continue
            fps = 0.8 * fps + 0.2 / max(now - last_detect, 1e-3)
            last_detect = now

            ts_ms = max(int(now * 1000), last_ts_ms + 1)   # strictly increasing
            last_ts_ms = ts_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = hands.detect_for_video(img, ts_ms)
            if res.hand_landmarks:
                g, gx, gy = _classify(res.hand_landmarks[0], g)
                x = float(np.clip(1.0 - gx, 0, 1))          # mirror
                y = float(np.clip(gy, 0, 1))
                last_seen = now
            elif now - last_seen > 0.3:         # ride through brief dropouts
                g = None
            if state_q.qsize() < 4:
                state_q.put(("state", x, y, g, fps))
        cap.release()
    except Exception as e:                                  # never die silently
        import traceback
        state_q.put(("error", f"{e}\n{traceback.format_exc()}"))
