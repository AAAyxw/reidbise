import logging
import os

import numpy as np

_log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
GUI_DIR = os.path.join(PROJECT_ROOT, "GUI")
MODELS_DIR = os.path.join(GUI_DIR, "models")
# log file setting
SAVE_LOG_PATH = os.path.join(GUI_DIR, 'outputs', 'logs')
# setting of YOLO model
YOLO_LABELS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
YOLO_MODEL_PATH_V8 = os.path.join(MODELS_DIR, 'yolov8s.onnx')
YOLO_MODEL_PATH_V12 = os.path.join(MODELS_DIR, 'yolov12s.pt')
YOLO_MODEL_VERSION = os.getenv('YOLO_MODEL_VERSION', 'v12').lower()
YOLO_MODEL_PATH = YOLO_MODEL_PATH_V12 if YOLO_MODEL_VERSION == 'v12' else YOLO_MODEL_PATH_V8
YOLO_DEFAULT_LABEL = [0] # 0 is 'person'
YOLO_MIN_SIZE = 0
YOLO_TRACKER_TYPE = 'botsort.yaml' # "bytetrack.yaml"

# setting of reid model
EXTRACTOR_PERSON = os.path.join(MODELS_DIR, 'reid_person_0.737.onnx')
EXTRACTOR_CROSS_MODAL = os.path.join(MODELS_DIR, 'reid_cross_modal.onnx')
REID_IN_SIZE = [256, 128]

# cross-modal (visible + infrared) re-id
CROSS_MODAL_ENABLED = os.getenv('CROSS_MODAL_ENABLED', '1') == '1'
CROSS_MODAL_FUSION = os.getenv('CROSS_MODAL_FUSION', 'average')
FUSION_WEIGHT_VISIBLE = 0.5
FUSION_WEIGHT_IR = 0.5


def _onnx_input_hw(input_meta, default_h=256, default_w=128):
    shape = input_meta.shape
    height = int(shape[2]) if len(shape) > 2 and isinstance(shape[2], int) else default_h
    width = int(shape[3]) if len(shape) > 3 and isinstance(shape[3], int) else default_w
    return height, width


def probe_onnx_feature_dim(model_path, num_inputs=1, in_size=None):
    """
    Run a dummy forward pass and return feature length (same slicing as ReIdExtract: output[0][0]).
    """
    if not model_path or not os.path.exists(model_path):
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        _log.warning("onnxruntime not available; cannot probe %s", model_path)
        return None

    default_h, default_w = (in_size or REID_IN_SIZE)
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as exc:
        _log.warning("Failed to open ONNX for dim probe %s: %s", model_path, exc)
        return None

    inputs = session.get_inputs()
    if len(inputs) < num_inputs:
        _log.warning(
            "Model %s has %d inputs, expected %d for dim probe",
            model_path, len(inputs), num_inputs,
        )
        return None

    feed = {}
    for idx in range(num_inputs):
        inp = inputs[idx]
        height, width = _onnx_input_hw(inp, default_h, default_w)
        feed[inp.name] = np.zeros((1, 3, height, width), dtype=np.float32)

    try:
        output = session.run(None, feed)[0]
    except Exception as exc:
        _log.warning("Failed to probe feature dim for %s: %s", model_path, exc)
        return None

    feature = np.asarray(output)[0]
    if feature.ndim > 1:
        feature = feature.reshape(-1)
    dim = int(feature.shape[0])
    if dim <= 0:
        return None
    return dim


def resolve_active_extractor_path(onnx_model=None):
    """Same model path resolution as CrossModalReIdExtract / ReidPipeline."""
    if onnx_model is not None:
        return onnx_model
    if CROSS_MODAL_ENABLED and os.path.exists(EXTRACTOR_CROSS_MODAL):
        return EXTRACTOR_CROSS_MODAL
    return EXTRACTOR_PERSON


def _onnx_num_inputs(model_path):
    if not model_path or not os.path.exists(model_path):
        return 0
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return len(session.get_inputs())
    except Exception:
        return 0


def resolve_reid_feature_dims(
    onnx_model=None,
    cross_modal_enabled=None,
    fusion_mode=None,
    extractor_person=None,
    extractor_cross_modal=None,
):
    """
    Resolve FAISS feature dimension to match build_extractor(onnx_model=...) output.
    ReidPipeline passes EXTRACTOR_PERSON explicitly, so default onnx_model matches that.
    """
    cross_modal_enabled = CROSS_MODAL_ENABLED if cross_modal_enabled is None else cross_modal_enabled
    fusion_mode = (fusion_mode or CROSS_MODAL_FUSION or 'average').lower()
    extractor_person = extractor_person or EXTRACTOR_PERSON
    extractor_cross_modal = extractor_cross_modal or EXTRACTOR_CROSS_MODAL
    active_path = resolve_active_extractor_path(onnx_model)

    person_dim = probe_onnx_feature_dim(extractor_person, num_inputs=1)

    if not cross_modal_enabled:
        if person_dim is not None:
            return person_dim
        return int(os.getenv('REID_DIMS_FALLBACK', '1280'))

    active_is_cross = os.path.abspath(active_path) == os.path.abspath(extractor_cross_modal)
    if active_is_cross and os.path.exists(extractor_cross_modal):
        if _onnx_num_inputs(extractor_cross_modal) >= 2:
            cross_dim = probe_onnx_feature_dim(extractor_cross_modal, num_inputs=2)
            if cross_dim is not None:
                return cross_dim

        cross_dim = probe_onnx_feature_dim(extractor_cross_modal, num_inputs=1)
        if cross_dim is not None:
            if fusion_mode == 'concat':
                return cross_dim * 2
            return cross_dim

    # Late fusion on single-stream ONNX (ReidPipeline default: reid_person)
    if person_dim is not None:
        if fusion_mode == 'concat':
            return person_dim * 2
        return person_dim

    return int(os.getenv('REID_DIMS_FALLBACK', '1280'))


def resolve_reid_single_dim(extractor_person=None):
    """Single-stream (visible-only) feature dimension."""
    extractor_person = extractor_person or EXTRACTOR_PERSON
    dim = probe_onnx_feature_dim(extractor_person, num_inputs=1)
    if dim is not None:
        return dim
    return int(os.getenv('REID_DIMS_FALLBACK', '1280'))


# ReidPipeline always passes EXTRACTOR_PERSON into build_extractor today.
PIPELINE_EXTRACTOR_PATH = EXTRACTOR_PERSON

DIMS_SINGLE = resolve_reid_single_dim()
DIMS = resolve_reid_feature_dims(onnx_model=PIPELINE_EXTRACTOR_PATH)

_log.info(
    "ReID feature dims: DIMS_SINGLE=%s DIMS=%s extractor=%s (cross_modal=%s fusion=%s)",
    DIMS_SINGLE,
    DIMS,
    PIPELINE_EXTRACTOR_PATH,
    CROSS_MODAL_ENABLED,
    CROSS_MODAL_FUSION,
)

# setting of qt sql
DB_PATH = os.path.join(GUI_DIR, 'reid.db')
DB_NAME = 'reid'

# ReID match gates (L2 distance on L2-normalized features; cos_sim = 1 - dist^2/2)
MATCH_DIST_THRESH = float(os.getenv('MATCH_DIST_THRESH', '0.15'))
MATCH_MIN_MARGIN = float(os.getenv('MATCH_MIN_MARGIN', '0.06'))
MATCH_MIN_COSINE_SIM = float(os.getenv('MATCH_MIN_COSINE_SIM', '0.95'))
# Max L2 distance to gallery centroid when only one photo per ID is registered
GALLERY_SINGLE_RADIUS = float(os.getenv('GALLERY_SINGLE_RADIUS', '0.11'))
GALLERY_RADIUS_SCALE = float(os.getenv('GALLERY_RADIUS_SCALE', '1.15'))
# Only lock a video track ID after a high-confidence match (reduces wrong ID stickiness)
TRACK_LOCK_DIST_THRESH = float(os.getenv('TRACK_LOCK_DIST_THRESH', '0.12'))

# 添加结果保存相关配置
SAVE_MATCHES = True  # 是否保存匹配结果
SAVE_DIR = "outputs/matches"  # 匹配结果保存目录
SAVE_THRESHOLD = 0.3  # 保存阈值
MAX_MATCHES_PER_TARGET = 5  # 每个目标最多保存的匹配数量
