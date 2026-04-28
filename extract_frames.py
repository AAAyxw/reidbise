import cv2
import os

video_path = r"D:\reid\data\行人检测视频01.flv"
out_dir = r"D:\reidbise-yolov12\datasets\person_det\images\val"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25  # 兜底
step = int(round(fps))  # 每秒1帧

idx = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % step == 0:
        saved += 1
        cv2.imwrite(os.path.join(out_dir, f"{saved:06d}.jpg"), frame)
    idx += 1

cap.release()
print(f"抽帧完成，共保存 {saved} 张到: {out_dir}")