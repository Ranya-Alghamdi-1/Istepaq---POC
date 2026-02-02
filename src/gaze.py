'''import numpy as np

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362

LEFT_EYE_UP = 159
LEFT_EYE_DOWN = 145
RIGHT_EYE_UP = 386
RIGHT_EYE_DOWN = 374

NOSE_TIP = 1
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
FOREHEAD = 10

def _pt(landmarks, idx):
    p = landmarks[idx]
    return np.array([p[0], p[1]], dtype=np.float32)

def iris_center(landmarks, which="left"):
    ids = LEFT_IRIS if which == "left" else RIGHT_IRIS
    pts = np.stack([_pt(landmarks, i) for i in ids], axis=0)
    return pts.mean(axis=0)

def eye_box_norm(landmarks, which="left"):
    if which == "left":
        outer = _pt(landmarks, LEFT_EYE_OUTER)
        inner = _pt(landmarks, LEFT_EYE_INNER)
        up = _pt(landmarks, LEFT_EYE_UP)
        down = _pt(landmarks, LEFT_EYE_DOWN)
    else:
        outer = _pt(landmarks, RIGHT_EYE_OUTER)
        inner = _pt(landmarks, RIGHT_EYE_INNER)
        up = _pt(landmarks, RIGHT_EYE_UP)
        down = _pt(landmarks, RIGHT_EYE_DOWN)

    w = np.linalg.norm(inner - outer) + 1e-6
    h = np.linalg.norm(up - down) + 1e-6
    return outer, inner, up, down, w, h

def iris_position_normalized(landmarks, which="left"):
    c = iris_center(landmarks, which)
    outer, inner, up, down, w, h = eye_box_norm(landmarks, which)

    x = np.dot(c - outer, (inner - outer) / w)
    x = np.clip(x, 0.0, w)

    y = np.dot(c - up, (down - up) / h)
    y = np.clip(y, 0.0, h)

    return np.array([x / w, y / h], dtype=np.float32)

def rough_head_orientation(landmarks):
    nose = _pt(landmarks, NOSE_TIP)
    lc = _pt(landmarks, LEFT_CHEEK)
    rc = _pt(landmarks, RIGHT_CHEEK)
    chin = _pt(landmarks, CHIN)
    forehead = _pt(landmarks, FOREHEAD)

    mid = (lc + rc) / 2.0
    half_width = np.linalg.norm(rc - lc) / 2.0 + 1e-6
    yaw = (nose[0] - mid[0]) / half_width

    face_height = np.linalg.norm(chin - forehead) + 1e-6
    pitch = (nose[1] - (forehead[1] + chin[1]) / 2.0) / (face_height / 2.0)

    return float(np.clip(yaw, -1.0, 1.0)), float(np.clip(pitch, -1.0, 1.0))

def is_looking_at_camera(landmarks, iris_center_tol, yaw_tol, pitch_tol):
    yaw, pitch = rough_head_orientation(landmarks)

    left_xy = iris_position_normalized(landmarks, "left")
    right_xy = iris_position_normalized(landmarks, "right")

    left_ok = (abs(left_xy[0] - 0.5) < iris_center_tol) and (abs(left_xy[1] - 0.5) < iris_center_tol)
    right_ok = (abs(right_xy[0] - 0.5) < iris_center_tol) and (abs(right_xy[1] - 0.5) < iris_center_tol)

    head_ok = (abs(yaw) < yaw_tol) and (abs(pitch) < pitch_tol)

    return bool(head_ok and left_ok and right_ok), (yaw, pitch), (left_xy, right_xy)'''
import numpy as np

# MediaPipe FaceMesh landmark indices (approx)
# Left eye corners: 33 (outer), 133 (inner)
# Right eye corners: 362 (outer), 263 (inner)
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263

# Iris landmarks (refine_landmarks must be enabled normally for iris;
# BUT even with refine_landmarks=False, many setups still provide stable eye area.
# We'll use eye corners + central eye points approximation.
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473


def _safe_point(lm: np.ndarray, idx: int):
    if idx < 0 or idx >= lm.shape[0]:
        return None
    return lm[idx]


def is_looking_at_camera(lm: np.ndarray, iris_center_tol=0.22, yaw_tol=0.30, pitch_tol=0.30):
    """
    Heuristic:
    - estimate iris horizontal position relative to eye corners.
    - if iris is near center for both eyes => looking at camera.
    Head yaw/pitch in this POC is not true 3D; kept as placeholders.
    """
    l_outer = _safe_point(lm, LEFT_EYE_OUTER)
    l_inner = _safe_point(lm, LEFT_EYE_INNER)
    r_outer = _safe_point(lm, RIGHT_EYE_OUTER)
    r_inner = _safe_point(lm, RIGHT_EYE_INNER)

    l_iris = _safe_point(lm, LEFT_IRIS_CENTER)
    r_iris = _safe_point(lm, RIGHT_IRIS_CENTER)

    if any(p is None for p in [l_outer, l_inner, r_outer, r_inner, l_iris, r_iris]):
        return False, (0.0, 0.0), {}

    # Horizontal ratio within eye (0..1)
    l_ratio = (l_iris[0] - l_outer[0]) / max(1e-6, (l_inner[0] - l_outer[0]))
    r_ratio = (r_iris[0] - r_outer[0]) / max(1e-6, (r_inner[0] - r_outer[0]))

    # Centered if close to 0.5
    l_centered = abs(l_ratio - 0.5) <= iris_center_tol
    r_centered = abs(r_ratio - 0.5) <= iris_center_tol

    looking = l_centered and r_centered

    # Placeholder head angles (not used in this demo)
    yaw = 0.0
    pitch = 0.0

    debug = {"l_ratio": float(l_ratio), "r_ratio": float(r_ratio)}
    return looking, (yaw, pitch), debug

