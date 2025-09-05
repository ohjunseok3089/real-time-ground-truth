import os

VIDEO_DEVICE = "/dev/video10"
USE_GPU = True
YOLO_MODEL = "yolo11n.pt"
SHOW_WINDOW = bool(os.getenv("SHOW_WINDOW", "False")) # ssh env is auto-disable
DRAW_YOLO = True
DRAW_FACE = True

# Person Recognition Settings
ENABLE_PERSON_RECOGNITION = True
MAX_EMBEDDINGS = 36000
RECOGNITION_THRESHOLD = 0.6