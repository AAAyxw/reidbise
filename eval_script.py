import os
import time
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from ultralytics import YOLO

# ================= 配置区 =================
# 300张原始图片所在的目录
IMG_DIR = r"D:\reidbise-yolov12\datasets\person_det\images\val" 
# 100个【人工修订后】的XML标注所在的目录
XML_DIR = r"D:\reidbise-yolov12\datasets\person_det\labels\val"      

MODEL_V12_PATH = r"D:\reidbise-yolov12\person_reid\GUI\models\yolov12s.pt"
MODEL_V8_ONNX_PATH = r"D:\reidbise-yolov12\person_reid\GUI\models\yolov8s.onnx"

IMG_SIZE = 640               # 统一推理尺寸
CONF_THRES = 0.25            # 置信度阈值
IOU_EVAL_THRES = 0.5         # 判定检测成功的IoU阈值
# ==========================================

def parse_xml_to_list(xml_path):
    """解析PASCAL VOC XML文件，返回归一化的 [xc, yc, w, h]"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        gt_boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name == 'person': 
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # 转换为 YOLO 格式 (中心点x, 中心点y, 宽, 高) 并归一化
                xc = (xmin + xmax) / 2 / width
                yc = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                gt_boxes.append([xc, yc, w, h])
        return np.array(gt_boxes)
    except Exception as e:
        print(f"解析XML出错 {xml_path}: {e}")
        return np.array([])

def calculate_iou(box1, box2):
    """计算两个 YOLO 格式框 [xc, yc, w, h] 的 IoU"""
    def to_corners(box):
        xc, yc, w, h = box
        return xc - w/2, yc - h/2, xc + w/2, yc + h/2

    b1_x1, b1_y1, b1_x2, b1_y2 = to_corners(box1)
    b2_x1, b2_y1, b2_x2, b2_y2 = to_corners(box2)

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter_area / (b1_area + b2_area - inter_area + 1e-6)

# 加载模型 (统一使用 Ultralytics 加载以确保前后处理一致)
print("正在加载模型...")
model_v12 = YOLO(MODEL_V12_PATH)
model_v8 = YOLO(MODEL_V8_ONNX_PATH) # 这样会自动调用 onnxruntime 且处理好NMS

def run_eval():
    # 以 XML 目录为准，确保只测试人工修订过的 100 张
    xml_files = [f for f in os.listdir(XML_DIR) if f.lower().endswith('.xml')]
    results_list = []

    print(f"检测到 {len(xml_files)} 个标注文件，开始对比测试...")

    for xml_name in xml_files:
        xml_path = os.path.join(XML_DIR, xml_name)
        
        # 匹配图片名（尝试多种后缀）
        base_name = os.path.splitext(xml_name)[0]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            tmp_path = os.path.join(IMG_DIR, base_name + ext)
            if os.path.exists(tmp_path):
                img_path = tmp_path
                break
        
        if img_path is None:
            print(f"跳过: 找不到对应的图片 {base_name}")
            continue

        # 1. 获取人工标注 (Ground Truth)
        gt_boxes = parse_xml_to_list(xml_path)
        if len(gt_boxes) == 0: continue

        img = cv2.imread(img_path)
        
        # --- YOLOv12s 推理 ---
        t1 = time.time()
        res_v12 = model_v12.predict(img, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)[0]
        v12_time = (time.time() - t1) * 1000
        v12_boxes = res_v12.boxes.xywhn.cpu().numpy()

        # --- YOLOv8.onnx 推理 ---
        t2 = time.time()
        res_v8 = model_v8.predict(img, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)[0]
        v8_time = (time.time() - t2) * 1000
        v8_boxes = res_v8.boxes.xywhn.cpu().numpy()

        # 计算召回率 (Recall) 和 准确率 (Precision) 逻辑
        def get_stats(pred_boxes, gt_boxes):
            hits = 0
            for gb in gt_boxes:
                if any(calculate_iou(gb, pb) > IOU_EVAL_THRES for pb in pred_boxes):
                    hits += 1
            recall = hits / len(gt_boxes) if len(gt_boxes) > 0 else 0
            precision = hits / len(pred_boxes) if len(pred_boxes) > 0 else 0
            return recall, precision

        v12_rec, v12_pre = get_stats(v12_boxes, gt_boxes)
        v8_rec, v8_pre = get_stats(v8_boxes, gt_boxes)
        
        results_list.append({
            'v12_recall': v12_rec, 'v12_precision': v12_pre, 'v12_ms': v12_time,
            'v8_recall': v8_rec, 'v8_precision': v8_pre, 'v8_ms': v8_time,
            'gt_count': len(gt_boxes)
        })

    # 输出结果
    df = pd.DataFrame(results_list)
    print("\n" + "="*50)
    print(f"测试完成！有效样本数: {len(df)}")
    summary = pd.DataFrame({
        'YOLOv12s (.pt)': [df['v12_recall'].mean(), df['v12_precision'].mean(), df['v12_ms'].mean()],
        'YOLOv8 (.onnx)': [df['v8_recall'].mean(), df['v8_precision'].mean(), df['v8_ms'].mean()]
    }, index=['平均召回率 (Recall)', '平均准确率 (Precision)', '平均推理延迟 (ms)'])
    
    print(summary.round(4))
    print("="*50)

if __name__ == "__main__":
    run_eval()