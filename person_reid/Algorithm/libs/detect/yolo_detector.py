import sys
import os
import numpy as np
from collections import defaultdict
import torch
from ultralytics import YOLO, __version__ as ultralytics_version
import Algorithm.libs.config.model_cfgs as cfgs
from Algorithm.libs.logger.log import get_logger

log_info = get_logger(__name__)

yolo_idx_names = cfgs.YOLO_LABELS


def _enable_torch26_ultralytics_compat():
    """
    PyTorch 2.6 defaults torch.load(weights_only=True). Old ultralytics
    checkpoints often require pickled objects, so force legacy behavior.
    """
    try:
        original_torch_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat
    except Exception as e:
        log_info.warning("skip torch.load compatibility patch: %s", e)


def _ensure_v12_runtime():
    """
    yolov12 weights include newer module definitions (e.g. C3k2).
    Ensure installed ultralytics has these symbols.
    """
    try:
        from ultralytics.nn.modules import block as yolo_block
        if not hasattr(yolo_block, "C3k2"):
            raise RuntimeError(
                "Installed ultralytics does not support YOLOv12 weights. "
                "Please upgrade ultralytics to a newer version."
            )
    except Exception as e:
        raise RuntimeError(
            "YOLOv12 runtime check failed with ultralytics=={}: {}"
            .format(ultralytics_version, e)
        )


class YoloDetect(object):
    def __init__(self, model_path = cfgs.YOLO_MODEL_PATH):
        _enable_torch26_ultralytics_compat()
        _ensure_v12_runtime()
        self._model = YOLO(model_path)
        self._model_track = YOLO(model_path)
        self._active_model_path = model_path
        log_info.info('%s model load succeed with ultralytics==%s', model_path, ultralytics_version)
    
#行人检测
    def detect(self, img, class_idx_list=cfgs.YOLO_DEFAULT_LABEL, min_size = cfgs.YOLO_MIN_SIZE):
        boxes, clss  = [], []
        results = self._model.predict(img, conf=0.4, iou=0.5, classes=class_idx_list)
        if results[0].boxes.xyxy is not None:
            _boxes = results[0].boxes.xyxy.cpu().tolist()
            _clss = results[0].boxes.cls.int().cpu().tolist()
            for _box, _cls in zip(_boxes, _clss):
                if _box[3] - _box[1] > min_size and _box[2] - _box[0] >min_size:
                    boxes.append(_box)
                    clss.append(_cls)
        return boxes, clss
        
    def track(self, frame, class_idx_list=cfgs.YOLO_DEFAULT_LABEL, persist=False,  min_size = cfgs.YOLO_MIN_SIZE, tracker=cfgs.YOLO_TRACKER_TYPE):
        boxes, track_ids, clss  = [], [], []
        results = self._model_track.track(frame, persist=persist, tracker=tracker, conf=0.2, iou=0.4,classes=class_idx_list)
        if results[0].boxes.id is not None:
            _boxes = results[0].boxes.xyxy.cpu().tolist()
            _track_ids = results[0].boxes.id.int().cpu().tolist()
            _clss = results[0].boxes.cls.int().cpu().tolist()
            for _box, _track_id, _cls in zip(_boxes, _track_ids, _clss):
                if _box[3] - _box[1] > min_size and _box[2] - _box[0] > min_size:
                    boxes.append(_box)
                    track_ids.append(_track_id)
                    clss.append(_cls)
        return boxes, track_ids, clss

    def reset_track(self):
        self._model_track = YOLO(self._active_model_path)