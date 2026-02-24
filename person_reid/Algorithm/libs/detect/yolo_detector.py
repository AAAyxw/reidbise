import sys
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import Algorithm.libs.config.model_cfgs as cfgs
from Algorithm.libs.logger.log import get_logger

log_info = get_logger(__name__)

yolo_idx_names = cfgs.YOLO_LABELS


class YoloDetect(object):
    def __init__(self, model_path = cfgs.YOLO_MODEL_PATH):
        self._model = YOLO(model_path)
        self._model_track = YOLO(model_path)
        log_info.info('{} model load succeed!!!'.format(model_path))
    
    def detect(self, img, class_idx_list=cfgs.YOLO_DEFAULT_LABEL, min_size = cfgs.YOLO_MIN_SIZE):
        boxes, clss  = [], []
        results = self._model.predict(img, conf=0.2, iou=0.4, classes=class_idx_list)
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
        self._model_track = YOLO(cfgs.YOLO_MODEL_PATH)