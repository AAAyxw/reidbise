
# log file setting
SAVE_LOG_PATH = './outputs/logs/'
# setting of YOLO model
YOLO_LABELS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
YOLO_MODEL_PATH_V8 = './models/yolov8s.onnx'
YOLO_MODEL_PATH_V12 = './models/yolov12s.pt'
YOLO_MODEL_VERSION = os.getenv('YOLO_MODEL_VERSION', 'v8').lower()
YOLO_MODEL_PATH = YOLO_MODEL_PATH_V12 if YOLO_MODEL_VERSION == 'v12' else YOLO_MODEL_PATH_V8
YOLO_DEFAULT_LABEL = [0] # 0 is 'person'
YOLO_MIN_SIZE = 0
YOLO_TRACKER_TYPE = 'botsort.yaml' # "bytetrack.yaml"

# setting of reid model
EXTRACTOR_PERSON = './models/reid_person_0.737.onnx'
REID_IN_SIZE = [256, 128]
DIMS = 1280

# setting of qt sql
DB_PATH = './reid.db'
DB_NAME = 'reid'

# 添加结果保存相关配置
SAVE_MATCHES = True  # 是否保存匹配结果
SAVE_DIR = "outputs/matches"  # 匹配结果保存目录
SAVE_THRESHOLD = 0.3  # 保存阈值
MAX_MATCHES_PER_TARGET = 5  # 每个目标最多保存的匹配数量