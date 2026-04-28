from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO(r"D:\reidbise-yolov12\person_reid\GUI\models\yolov12s.pt")  # 或 yolov8.onnx
img_dir = Path(r"D:\reidbise-yolov12\datasets\person_det\images\val")
lab_dir = Path(r"D:\reidbise-yolov12\datasets\person_det\labels\val")
lab_dir.mkdir(parents=True, exist_ok=True)

imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

for img_path in imgs:
    im = cv2.imread(str(img_path))
    h, w = im.shape[:2]
    r = model.predict(im, conf=0.25, iou=0.5, classes=[0], verbose=False)[0]  # 只person类
    txt_path = lab_dir / f"{img_path.stem}.txt"

    lines = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c in zip(xyxy, cls):
            # 转YOLO格式
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

print(f"自动标注完成，共处理 {len(imgs)} 张")