import cv2
import numpy as np
import onnxruntime as ort
import torch
import os
import torchvision.transforms as T
from PIL import Image
import Algorithm.libs.config.model_cfgs as cfgs
from Algorithm.libs.extract.cross_modal_utils import ensure_bgr
from Algorithm.libs.logger.log import get_logger

log_info = get_logger(__name__)


class ReIdExtract(object):
    def __init__(self, extract_class, onnx_model=cfgs.EXTRACTOR_PERSON, IN_SIZE=cfgs.REID_IN_SIZE, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        self._extract_class = extract_class
        self._onnx_model = onnx_model
        if not os.path.exists(self._onnx_model):
            raise FileNotFoundError(
                "ReID model file not found: {}. Please place reid_person_0.737.onnx in GUI/models/."
                .format(self._onnx_model)
            )
        self.session = ort.InferenceSession(self._onnx_model, providers=providers)
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        self.transform = T.Compose([
            T.Resize(IN_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        log_info.info("{} model loaded!!! The shape is {}_{}.".format(onnx_model, self.input_width, self.input_height))

    def _to_tensor(self, image_data):
        image_data = ensure_bgr(image_data)
        image_data = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        return self.transform(image_data)

    def _run_onnx(self, input_var):
        outputs = self.session.run(None, {self.model_inputs[0].name: input_var.numpy()})
        return outputs[0][0]

    def __call__(self, image_data, norm_feat=True):
        img = self._to_tensor(image_data)
        input_var = torch.stack([img], dim=0)
        features = self._run_onnx(input_var)
        if norm_feat:
            features = features / np.linalg.norm(features)
        return features


class CrossModalReIdExtract(ReIdExtract):
    """
    Dual-channel extractor for visible + infrared person re-id.
    Supports: (1) two-input ONNX, (2) late fusion with single-stream ONNX.
    """

    def __init__(self, extract_class, onnx_model=None, ir_onnx_model=None, IN_SIZE=cfgs.REID_IN_SIZE, providers=None):
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        cross_modal_path = cfgs.EXTRACTOR_CROSS_MODAL
        if onnx_model is None:
            onnx_model = cross_modal_path if os.path.exists(cross_modal_path) else cfgs.EXTRACTOR_PERSON
        super().__init__(extract_class, onnx_model, IN_SIZE, providers)
        self.fusion_mode = cfgs.CROSS_MODAL_FUSION
        self._dual_input_onnx = len(self.model_inputs) >= 2
        self._ir_session = None
        if ir_onnx_model and os.path.exists(ir_onnx_model):
            self._ir_session = ort.InferenceSession(ir_onnx_model, providers=providers)
        if self._dual_input_onnx:
            log_info.info("Cross-modal ONNX with %d inputs loaded.", len(self.model_inputs))
        else:
            log_info.info("Cross-modal late fusion mode=%s on %s", self.fusion_mode, onnx_model)

    def _run_dual_input_onnx(self, visible_img, ir_img, norm_feat=True):
        vis_tensor = self._to_tensor(visible_img)
        ir_tensor = self._to_tensor(ensure_bgr(ir_img))
        vis_var = torch.stack([vis_tensor], dim=0).numpy()
        ir_var = torch.stack([ir_tensor], dim=0).numpy()
        feed = {
            self.model_inputs[0].name: vis_var,
            self.model_inputs[1].name: ir_var,
        }
        features = self.session.run(None, feed)[0][0]
        if norm_feat:
            features = features / np.linalg.norm(features)
        return features

    def fuse_features(self, feat_vis, feat_ir, norm_feat=True):
        if self.fusion_mode == 'concat':
            fused = np.concatenate([feat_vis, feat_ir], axis=0)
        else:
            fused = (
                cfgs.FUSION_WEIGHT_VISIBLE * feat_vis
                + cfgs.FUSION_WEIGHT_IR * feat_ir
            )
        if norm_feat:
            norm = np.linalg.norm(fused)
            if norm > 1e-12:
                fused = fused / norm
        return fused

    def extract_visible(self, visible_img, norm_feat=True):
        return self.__call__(visible_img, ir_img=None, norm_feat=norm_feat)

    def extract_ir(self, ir_img, norm_feat=True):
        ir_img = ensure_bgr(ir_img)
        if self._ir_session is not None:
            # 使用独立的红外分支模型
            img = self._to_tensor(ir_img)
            input_var = torch.stack([img], dim=0)
            features = self._ir_session.run(None, {self._ir_session.get_inputs()[0].name: input_var.numpy()})[0][0]
        elif self._dual_input_onnx:
            # 对于双通道模型，使用红外图像作为两个输入
            return self._run_dual_input_onnx(ir_img, ir_img, norm_feat=norm_feat)
        else:
            # 使用单通道模型
            features = self._run_onnx(torch.stack([self._to_tensor(ir_img)], dim=0))
        if norm_feat:
            features = features / np.linalg.norm(features)
        return features

    def __call__(self, visible_img, ir_img=None, norm_feat=True):
        visible_img = ensure_bgr(visible_img)
        if ir_img is None:
            # 当只有可见光输入时，使用红外图像作为占位符（跨模态场景）
            # 或者如果模型支持单输入，则使用单输入模式
            if self._dual_input_onnx:
                # 对于双通道模型，使用相同的图像作为两个输入（降级方案）
                return self._run_dual_input_onnx(visible_img, visible_img, norm_feat=norm_feat)
            else:
                return super().__call__(visible_img, norm_feat=norm_feat)
        ir_img = ensure_bgr(ir_img)
        if self._dual_input_onnx:
            return self._run_dual_input_onnx(visible_img, ir_img, norm_feat=norm_feat)
        feat_vis = super().__call__(visible_img, norm_feat=True)
        feat_ir = self.extract_ir(ir_img, norm_feat=True)
        return self.fuse_features(feat_vis, feat_ir, norm_feat=norm_feat)


def build_extractor(extract_class, device_info, onnx_model=None):
    providers = ['CPUExecutionProvider']
    if "gpu" in device_info.lower():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if cfgs.CROSS_MODAL_ENABLED:
        return CrossModalReIdExtract(extract_class, onnx_model=onnx_model, providers=providers)
    return ReIdExtract(extract_class, onnx_model or cfgs.EXTRACTOR_PERSON, cfgs.REID_IN_SIZE, providers=providers)
