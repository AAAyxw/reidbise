import cv2
import numpy as np


def ensure_bgr(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def crop_bbox(image, bbox):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return image[y1:y2, x1:x2]


def align_ir_to_visible(ir_image, visible_shape):
    """Resize IR frame to visible resolution when cameras are co-registered by scale only."""
    ir_image = ensure_bgr(ir_image)
    vis_h, vis_w = visible_shape[:2]
    if ir_image.shape[0] == vis_h and ir_image.shape[1] == vis_w:
        return ir_image
    return cv2.resize(ir_image, (vis_w, vis_h), interpolation=cv2.INTER_LINEAR)


def resolve_ir_frame(visible_img, ir_img=None):
    if ir_img is None:
        return None
    return align_ir_to_visible(ir_img, visible_img.shape)
