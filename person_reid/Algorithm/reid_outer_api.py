# coding: utf-8
# @Author    : LittleAnt
# @Time      :
# @Descrip   : Please contact [wechat:cv_littleant] on WeChat for any questions.

import cv2
import os
import numpy as np
from Algorithm.libs.extract.reid_extract import build_extractor
from Algorithm.libs.extract.cross_modal_utils import crop_bbox, resolve_ir_frame
from Algorithm.libs.detect.yolo_detector import YoloDetect
from Algorithm.libs.search.search_engine import SearchEngine
import Algorithm.libs.config.model_cfgs as cfgs
from Algorithm.libs.logger.log import get_logger

log_info = get_logger(__name__)

colo_idx_names = cfgs.YOLO_LABELS


def Singleton(cls):
    _instance = {}

    def wrapper(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return wrapper


@Singleton
class ReidPipeline(object):
    """
    Pipeline to process reid (visible + optional infrared cross-modal).
    """

    def __init__(
        self,
        base_feat_lists,
        base_idx_lists,
        dims=None,
        target_class="person",
        device_info="cpu",
        modalities=None,
    ):
        self._target_class = target_class
        self._device_info = device_info
        self._dims = dims if dims is not None else cfgs.DIMS
        self._detector = YoloDetect(cfgs.YOLO_MODEL_PATH)
        self._search_engine = SearchEngine(
            base_feat_lists, base_idx_lists, dims=self._dims, modalities=modalities,
        )
        if base_feat_lists and len(base_feat_lists) > 0:
            self._dims = self._search_engine._dims
        self.track_method = cfgs.YOLO_TRACKER_TYPE
        self.cross_modal_enabled = cfgs.CROSS_MODAL_ENABLED
        if self._target_class == "person":
            self._input_size = cfgs.REID_IN_SIZE
            self._target_class_idx_list = [0]
            extractor_path = cfgs.PIPELINE_EXTRACTOR_PATH
        else:
            raise NotImplementedError("This function only support person reid")
        self._extractor = build_extractor(self._target_class, self._device_info, onnx_model=extractor_path)
        log_info.info(
            "ReidPipeline ready: cross_modal=%s dims=%s fusion=%s extractor=%s",
            self.cross_modal_enabled,
            self._dims,
            cfgs.CROSS_MODAL_FUSION,
            extractor_path,
        )

    def reload_reid_model(self, in_extractor_path=None, target_class=None, device=None):
        if target_class is not None:
            self._target_class = target_class
        if device is not None:
            self._device_info = device
        log_info.info(
            "!!!reload reid model, convert the model to {} and based on {}".format(
                self._target_class, self._device_info
            )
        )
        if self._target_class == "person":
            self._input_size = cfgs.REID_IN_SIZE
            self._target_class_idx_list = [0]
            extractor_path = cfgs.PIPELINE_EXTRACTOR_PATH
        else:
            raise NotImplementedError("This function only support person reid")
        if in_extractor_path is not None:
            extractor_path = in_extractor_path
        self._extractor = build_extractor(self._target_class, self._device_info, onnx_model=extractor_path)
        self._dims = cfgs.resolve_reid_feature_dims(onnx_model=extractor_path)

    def reload_search_engine(self, base_feat_lists, base_idx_lists, dims=None, modalities=None):
        log_info.info("!!!reload faiss search engine")
        if base_feat_lists and len(base_feat_lists) > 0:
            gallery_dim = len(base_feat_lists[0])
            if dims is not None and dims != gallery_dim:
                log_info.warning(
                    "Configured dims %s != gallery dims %s; using gallery dims.",
                    dims, gallery_dim,
                )
            dims = gallery_dim
        elif dims is None:
            dims = self._dims
        self._dims = dims
        self._search_engine = SearchEngine(
            base_feat_lists, base_idx_lists, dims=dims, modalities=modalities,
        )
        if self._search_engine._index is not None:
            self._dims = self._search_engine._dims

    def detect(self, img, class_idx_list, format='image', is_track=False):
        if format == 'image':
            boxes, clss = self._detector.detect(img, class_idx_list=class_idx_list)
            track_ids = None
        elif format == 'video':
            if is_track:
                boxes, track_ids, clss = self._detector.track(
                    img, class_idx_list, persist=True, tracker=self.track_method
                )
            else:
                boxes, clss = self._detector.detect(img, class_idx_list=class_idx_list)
                track_ids = None
        return boxes, track_ids, clss

    def extract(self, vis_img, ir_img=None):
        aligned_ir = resolve_ir_frame(vis_img, ir_img)
        if self.cross_modal_enabled and aligned_ir is not None:
            return self._extractor(vis_img, ir_img=aligned_ir)
        return self._extractor(vis_img)

    def _extract_crop(self, vis_img, ir_img, bbox):
        try:
            vis_crop = crop_bbox(vis_img, bbox)
            if vis_crop is None or vis_crop.size == 0:
                log_info.warning("Empty visible crop, skipping bbox: {}".format(bbox))
                return None
            
            if ir_img is None:
                return self._extractor(vis_crop)
            
            ir_aligned = resolve_ir_frame(vis_img, ir_img)
            if ir_aligned is None:
                log_info.warning("Failed to align IR frame, using visible only")
                return self._extractor(vis_crop)
            
            ir_crop = crop_bbox(ir_aligned, bbox)
            if ir_crop is None or ir_crop.size == 0:
                log_info.warning("Empty IR crop, using visible only")
                return self._extractor(vis_crop)
            
            if self.cross_modal_enabled:
                return self._extractor(vis_crop, ir_img=ir_crop)
            return self._extractor(vis_crop)
        except Exception as e:
            log_info.error("Error extracting crop: {}".format(str(e)))
            return None

    def search(self, vis_img, bboxs, thresh=None, ir_img=None, query_modality=None):
        if thresh is None:
            thresh = cfgs.MATCH_DIST_THRESH
        search_labels_list, search_dist_list = [], []
        before_sort_list = []
        before_dist_list = []
        filter_box_list = []
        ir_aligned = resolve_ir_frame(vis_img, ir_img)
        if query_modality is None:
            query_modality = "vis_ir" if (self.cross_modal_enabled and ir_aligned is not None) else "vis"
        for bbox in bboxs:
            _each_img_norm_feat = self._extract_crop(vis_img, ir_aligned, bbox)
            if _each_img_norm_feat is None:
                log_info.warning("Failed to extract feature for bbox: {}".format(bbox))
                before_sort_list.append("unknown")
                before_dist_list.append(None)
                continue
            search_labels, search_dist = self._search_engine.search(_each_img_norm_feat, 10)
            if len(search_labels) > 0 and len(search_dist) > 0:
                search_labels, search_dist = self._search_engine.rerank(
                    _each_img_norm_feat, search_labels, search_dist, query_modality,
                )
            elif len(search_labels) == 0 and self._search_engine._index is None:
                log_info.warning("Search index is empty; register targets in the gallery first.")

            accepted, match_label, match_dist = self._search_engine.decide_match(
                search_labels, search_dist, thresh, query_feat=_each_img_norm_feat,
            )
            if not accepted:
                before_sort_list.append("unknown")
                before_dist_list.append(match_dist)
                continue
            search_labels_list.append(match_label)
            search_dist_list.append(float(match_dist))
            filter_box_list.append(bbox)
            before_sort_list.append(match_label)
            before_dist_list.append(float(match_dist))
        return search_labels_list, search_dist_list, filter_box_list, before_sort_list, before_dist_list

    def reset_track(self):
        self._detector.reset_track()
