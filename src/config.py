'''CONFIG = {
    "CAMERA_INDEX": 0,
    "FRAME_WIDTH": 1280,
    "FRAME_HEIGHT": 720,

    # حساسية “النظر للكاميرا”
    "IRIS_CENTER_TOL": 0.22,   # 
    "HEAD_YAW_TOL": 0.25,
    "HEAD_PITCH_TOL": 0.25,

    # شرط الديمو الأساسي
    "MAX_NO_LOOK_SECONDS": 60.0,

    "SHOW_LANDMARKS": False,
}'''
'''CONFIG = {
    # Input source
    "USE_VIDEO_FILE": True,
    "VIDEO_PATH": "C:\\Users\\Raniy\\OneDrive\\Desktop\\Istepaq\\data\\raniyah_eyes.mp4",  # The video path here 
    #Drop before GitHub
    "CAMERA_INDEX": 0,

    # Demo threshold (fast)
    "DEMO_NO_LOOK_SECONDS": 8.0,   # for the 40s demo
    "PROD_NO_LOOK_SECONDS": 60.0,  # reference for real deployment

    # Performance
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 360,

    # Gaze sensitivity (tune if needed)
    "IRIS_CENTER_TOL": 0.22,
    "HEAD_YAW_TOL": 0.30,
    "HEAD_PITCH_TOL": 0.30,

    # UI
    "SHOW_LANDMARKS": False,
    "SUBJECT_LABEL": "Raniyah",
}'''
'''CONFIG = {
    "USE_VIDEO_FILE": True,
    "VIDEO_PATH": "data/raniyah_eyes.mp4",
    "CAMERA_INDEX": 0,

    "DEMO_NO_LOOK_SECONDS": 8.0,
    "PROD_NO_LOOK_SECONDS": 60.0,

    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 360,

    "IRIS_CENTER_TOL": 0.22,
    "HEAD_YAW_TOL": 0.30,
    "HEAD_PITCH_TOL": 0.30,

    # ✅ NEW: stabilize the gaze decision (reduces flicker)
    "HOLD_FRAMES": 10,

    # ✅ NEW: output files
    "OUTPUT_VIDEO": "outputs/istepaq_demo_output.mp4",
    "EVENTS_JSONL": "outputs/events.jsonl",
    "SUMMARY_JSON": "outputs/summary.json",

    "SUBJECT_LABEL": "Raniyah",
    "SHOW_LANDMARKS": False,
}'''
'''CONFIG = {
    "USE_VIDEO_FILE": True,
    "VIDEO_PATH": "data/raniyah_eyes.mp4",

    # ✅ أسرع للديمو
    "DEMO_NO_LOOK_SECONDS": 2.0,


    # جودة/أداء
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 360,

    "IRIS_CENTER_TOL": 0.22,
    "HEAD_YAW_TOL": 0.30,
    "HEAD_PITCH_TOL": 0.30,

    # ✅ يقلل الفلكر
    "HOLD_FRAMES": 10,

    # ✅ مخرجات فقط
    "OUTPUT_VIDEO": "outputs/istepaq_demo_output.mp4",
    "SUMMARY_JSON": "outputs/summary.json",

}'''
CONFIG = {
    "USE_VIDEO_FILE": True,
    "VIDEO_PATH": "data/raniyah_eyes.mp4",

    # أسرع اكتشاف مخالفة
    "DEMO_NO_LOOK_SECONDS": 2.0,
    "HOLD_FRAMES": 3,

    # حجم الفريم
    "FRAME_WIDTH": 640,
    "FRAME_HEIGHT": 360,

    # حساسية أعلى
    "IRIS_CENTER_TOL": 0.18,
    "HEAD_YAW_TOL": 0.25,
    "HEAD_PITCH_TOL": 0.25,

    #  المخرجات
    "OUTPUT_VIDEO": "outputs/istepaq_demo_output.mp4",
    "SUMMARY_JSON": "outputs/summary.json",

    # 👤 الاسم
    "SUBJECT_LABEL": "Raniyah"
}




