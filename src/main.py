'''import os
import json
import cv2
import numpy as np
import mediapipe as mp

from config import CONFIG
from utils import now, TimerState
from gaze import is_looking_at_camera


def draw_text(frame, text, x, y, color=(255, 255, 255), scale=0.8, thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_badge(frame, text, level="ok"):
    # ok / warn / danger
    if level == "ok":
        bg = (0, 170, 0)
    elif level == "warn":
        bg = (0, 170, 255)
    else:
        bg = (0, 0, 255)

    x1, y1 = 420, 15
    x2, y2 = 630, 55
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, text, (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)


def draw_progress_bar(frame, value, max_value, x=15, y=110, w=320, h=18):
    p = 0.0 if max_value <= 0 else min(value / max_value, 1.0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    fill_w = int(w * p)
    fill_color = (0, 0, 255) if p >= 1.0 else (0, 200, 0)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fill_color, -1)

    draw_text(frame, f"Attention timer: {value:.1f}/{max_value:.0f}s", x, y - 8, color=(255, 255, 255), scale=0.6, thickness=2)


def open_source():
    if CONFIG.get("USE_VIDEO_FILE", False):
        return cv2.VideoCapture(CONFIG["VIDEO_PATH"])
    return cv2.VideoCapture(CONFIG.get("CAMERA_INDEX", 0))


def make_writer(output_path: str, fps: float, frame_size: tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def main():
    cap = open_source()
    if not cap.isOpened():
        raise RuntimeError("Could not open video source. Check VIDEO_PATH in config.py")

    os.makedirs("outputs", exist_ok=True)

    out_video_path = CONFIG.get("OUTPUT_VIDEO", "outputs/istepaq_demo_output.mp4")
    summary_path = CONFIG.get("SUMMARY_JSON", "outputs/summary.json")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1:
        src_fps = 30.0

    frame_w = int(CONFIG.get("FRAME_WIDTH", 640))
    frame_h = int(CONFIG.get("FRAME_HEIGHT", 360))
    threshold = float(CONFIG.get("DEMO_NO_LOOK_SECONDS", 2.0))
    hold_frames = int(CONFIG.get("HOLD_FRAMES", 10))
    label = CONFIG.get("SUBJECT_LABEL", "Raniyah")

    out = make_writer(out_video_path, float(src_fps), (frame_w, frame_h))

    mp_face_mesh = mp.solutions.face_mesh
    no_look_timer = TimerState()

    # Hysteresis (stability)
    stable_true = 0
    stable_false = 0
    stable_looking = True

    # Summary counters
    total_frames = 0
    violation_count = 0
    active_violation_start = None
    violation_total_seconds = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1
            t = now()

            frame = cv2.resize(frame, (frame_w, frame_h))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(frame_rgb)

            # Clean header for judges
            draw_text(frame, "Istepaq - POC", 15, 30, color=(255, 255, 255), scale=0.85, thickness=2)
            draw_text(frame, f"Subject: {label}", 15, 60, color=(255, 255, 255), scale=0.85, thickness=2)

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                lm = np.array([[p.x, p.y] for p in face.landmark], dtype=np.float32)

                looking_raw, _, _ = is_looking_at_camera(
                    lm,
                    iris_center_tol=CONFIG.get("IRIS_CENTER_TOL", 0.22),
                    yaw_tol=CONFIG.get("HEAD_YAW_TOL", 0.30),
                    pitch_tol=CONFIG.get("HEAD_PITCH_TOL", 0.30)
                )

                # Hysteresis update
                if looking_raw:
                    stable_true += 1
                    stable_false = 0
                else:
                    stable_false += 1
                    stable_true = 0

                if stable_true >= hold_frames:
                    stable_looking = True
                elif stable_false >= hold_frames:
                    stable_looking = False

                no_look_seconds = no_look_timer.update(condition=(not stable_looking), t=t)

                # Badge logic
                if no_look_seconds >= threshold:
                    draw_badge(frame, "VIOLATION", level="danger")
                elif no_look_seconds >= threshold * 0.6:
                    draw_badge(frame, "WARNING", level="warn")
                else:
                    draw_badge(frame, "COMPLIANT", level="ok")

                draw_progress_bar(frame, no_look_seconds, threshold, x=15, y=110, w=320, h=18)

                # Violation counter + summary
                if no_look_seconds >= threshold and active_violation_start is None:
                    violation_count += 1
                    active_violation_start = t

                if no_look_seconds < threshold and active_violation_start is not None:
                    dur = float(t - active_violation_start)
                    violation_total_seconds += dur
                    active_violation_start = None

                # Show violation counter clearly
                draw_text(frame, f"Violations: {violation_count}", 15, 170, color=(255, 255, 255), scale=0.85, thickness=2)

                # Big alert text only when violating
                if no_look_seconds >= threshold:
                    cv2.rectangle(frame, (10, 210), (630, 255), (0, 0, 255), 2)
                    draw_text(frame, "ALERT: Not looking at camera", 18, 242, color=(0, 0, 255), scale=0.85, thickness=2)

            else:
                # Minimal message (no spam)
                draw_badge(frame, "NO FACE", level="warn")
                draw_text(frame, "Face not detected", 15, 120, color=(0, 170, 255), scale=0.75, thickness=2)
                no_look_timer.update(False, t)
                draw_text(frame, f"Violations: {violation_count}", 15, 170, color=(255, 255, 255), scale=0.85, thickness=2)

            out.write(frame)

            cv2.imshow("Istepaq Monitor - POC (Video Mode)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

    end_t = now()
    if active_violation_start is not None:
        violation_total_seconds += float(end_t - active_violation_start)
        active_violation_start = None

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    summary = {
        "project": "Istepaq Monitor POC",
        "subject": label,
        "video_input": CONFIG.get("VIDEO_PATH", None) if CONFIG.get("USE_VIDEO_FILE", False) else None,
        "output_video": out_video_path,
        "threshold_no_look_seconds": float(threshold),
        "hold_frames": int(hold_frames),
        "total_frames": int(total_frames),
        "fps_used": float(src_fps),
        "violation_count": int(violation_count),
        "violation_total_seconds": float(violation_total_seconds),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved output video:", out_video_path)
    print("✅ Summary (json):", summary_path, "\n")


if __name__ == "__main__":
    main()'''
import os
import json
import cv2
import numpy as np
import mediapipe as mp

from config import CONFIG
from utils import now, TimerState
from gaze import is_looking_at_camera


# ---------- UI Helpers ----------
def draw_text(frame, text, x, y, color=(255, 255, 255), scale=0.55, thickness=1):
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame, x, y, w, h, alpha=0.35):
    """Semi-transparent panel for clean info display."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_badge(frame, text, level="ok"):
    """Small top-right badge: COMPLIANT / WARNING / VIOLATION."""
    if level == "ok":
        bg = (0, 170, 0)
    elif level == "warn":
        bg = (0, 170, 255)
    else:
        bg = (0, 0, 255)

    x1, y1 = 470, 12
    x2, y2 = 630, 44
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(frame, text, (x1 + 10, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


'''def draw_progress_bar(frame, value, max_value, x=20, y=105, w=300, h=14):
    """Progress bar for attention timer (no-look duration)."""
    p = 0.0 if max_value <= 0 else min(value / max_value, 1.0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    fill_w = int(w * p)
    fill_color = (0, 0, 255) if p >= 1.0 else (0, 200, 0)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fill_color, -1)'''

def draw_progress_bar(frame, value, max_value, x=20, y=85, w=230, h=10, alpha=0.35):
    """Small semi-transparent progress bar."""
    p = 0.0 if max_value <= 0 else min(value / max_value, 1.0)

    overlay = frame.copy()

    # background (semi-transparent)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # fill
    fill_w = int(w * p)
    fill_color = (0, 0, 255) if p >= 1.0 else (0, 200, 0)
    cv2.rectangle(overlay, (x, y), (x + fill_w, y + h), fill_color, -1)

    # blend overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

# ---------- Video I/O ----------
def open_source():
    if CONFIG.get("USE_VIDEO_FILE", False):
        return cv2.VideoCapture(CONFIG["VIDEO_PATH"])
    return cv2.VideoCapture(CONFIG.get("CAMERA_INDEX", 0))


def make_writer(output_path: str, fps: float, frame_size: tuple[int, int]):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


# ---------- Main ----------
def main():
    cap = open_source()
    if not cap.isOpened():
        raise RuntimeError("Could not open video source. Check VIDEO_PATH in config.py")

    os.makedirs("outputs", exist_ok=True)

    out_video_path = CONFIG.get("OUTPUT_VIDEO", "outputs/istepaq_demo_output.mp4")
    summary_path = CONFIG.get("SUMMARY_JSON", "outputs/summary.json")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1:
        src_fps = 30.0

    frame_w = int(CONFIG.get("FRAME_WIDTH", 640))
    frame_h = int(CONFIG.get("FRAME_HEIGHT", 360))
    threshold = float(CONFIG.get("DEMO_NO_LOOK_SECONDS", 2.0))
    hold_frames = int(CONFIG.get("HOLD_FRAMES", 3))
    label = CONFIG.get("SUBJECT_LABEL", "Raniyah")

    out = make_writer(out_video_path, float(src_fps), (frame_w, frame_h))

    mp_face_mesh = mp.solutions.face_mesh
    no_look_timer = TimerState()

    # Stability (hysteresis)
    stable_true = 0
    stable_false = 0
    stable_looking = True

    # Summary counters
    total_frames = 0
    violation_count = 0
    active_violation_start = None
    violation_total_seconds = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1
            t = now()

            frame = cv2.resize(frame, (frame_w, frame_h))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(frame_rgb)

            # --- Top-left clean panel ---
            draw_panel(frame, x=10, y=10, w=320, h=95, alpha=0.35)
            draw_text(frame, "Istepaq - POC", 20, 35, scale=0.60, thickness=1)
            draw_text(frame, f"Subject: {label}", 20, 60, scale=0.60, thickness=1)
            draw_text(frame, f"Violations: {violation_count}", 20, 85, scale=0.60, thickness=1)

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                lm = np.array([[p.x, p.y] for p in face.landmark], dtype=np.float32)

                looking_raw, _, _ = is_looking_at_camera(
                    lm,
                    iris_center_tol=CONFIG.get("IRIS_CENTER_TOL", 0.18),
                    yaw_tol=CONFIG.get("HEAD_YAW_TOL", 0.25),
                    pitch_tol=CONFIG.get("HEAD_PITCH_TOL", 0.25)
                )

                # Hysteresis update
                if looking_raw:
                    stable_true += 1
                    stable_false = 0
                else:
                    stable_false += 1
                    stable_true = 0

                if stable_true >= hold_frames:
                    stable_looking = True
                elif stable_false >= hold_frames:
                    stable_looking = False

                no_look_seconds = no_look_timer.update(condition=(not stable_looking), t=t)

                # Badge logic
                if no_look_seconds >= threshold:
                    draw_badge(frame, "VIOLATION", level="danger")
                elif no_look_seconds >= threshold * 0.6:
                    draw_badge(frame, "WARNING", level="warn")
                else:
                    draw_badge(frame, "COMPLIANT", level="ok")

                # Progress bar
                #draw_progress_bar(frame, no_look_seconds, threshold, x=20, y=105, w=300, h=14)
                draw_progress_bar(frame, no_look_seconds, threshold, x=20, y=92, w=230, h=10, alpha=0.35)

                # Violation counter + summary timings
                if no_look_seconds >= threshold and active_violation_start is None:
                    violation_count += 1
                    active_violation_start = t

                if no_look_seconds < threshold and active_violation_start is not None:
                    dur = float(t - active_violation_start)
                    violation_total_seconds += dur
                    active_violation_start = None

                # Bottom alert (smaller, not intrusive)
                '''if no_look_seconds >= threshold:
                    y1 = frame_h - 55
                    y2 = frame_h - 15
                    cv2.rectangle(frame, (10, y1), (frame_w - 10, y2), (0, 0, 255), 1)
                    draw_text(frame, "ALERT: Not looking at camera",
                              20, frame_h - 28, color=(0, 0, 255), scale=0.65, thickness=2)'''
                if no_look_seconds >= threshold:
                    y1 = frame_h - 95
                    y2 = frame_h - 55
                    cv2.rectangle(frame, (10, y1), (frame_w - 10, y2), (0, 0, 255), 1)
                    draw_text(frame, "ALERT: Not looking at camera",
                       20, y2 - 12, color=(0, 0, 255), scale=0.60, thickness=2)

            else:
                draw_badge(frame, "NO FACE", level="warn")
                # keep message minimal
                draw_text(frame, "Face not detected", 20, 140, color=(0, 170, 255), scale=0.60, thickness=2)
                no_look_timer.update(False, t)

            # Write output + show
            out.write(frame)
            cv2.imshow("Istepaq Monitor - POC (Video Mode)", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

    end_t = now()
    if active_violation_start is not None:
        violation_total_seconds += float(end_t - active_violation_start)
        active_violation_start = None

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    summary = {
        "project": "Istepaq Monitor POC",
        "subject": label,
        "video_input": CONFIG.get("VIDEO_PATH", None) if CONFIG.get("USE_VIDEO_FILE", False) else None,
        "output_video": out_video_path,
        "threshold_no_look_seconds": float(threshold),
        "hold_frames": int(hold_frames),
        "total_frames": int(total_frames),
        "fps_used": float(src_fps),
        "violation_count": int(violation_count),
        "violation_total_seconds": float(violation_total_seconds),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved output video:", out_video_path)
    print("✅ Summary (json):", summary_path, "\n")


if __name__ == "__main__":
    main()


