import base64
import cv2
import traceback
import pdb
import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap

def save_match_image_with_info(image, save_path, similarity, additional_info=None):
    """保存带有相似度信息的匹配图像"""
    # 创建图像副本
    img_with_info = image.copy()
    
    # 添加相似度信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Similarity: {similarity:.3f}"
    
    # 获取文本大小
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    
    # 添加半透明背景
    overlay = img_with_info.copy()
    cv2.rectangle(overlay, (5, 5), 
                 (text_size[0] + 15, text_size[1] + 25), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img_with_info, 0.4, 0, img_with_info)
    
    # 添加文本
    cv2.putText(img_with_info, text, (10, text_size[1] + 15), 
                font, 0.7, (255, 255, 255), 2)
    
    # 如果有额外信息，添加到图像上
    if additional_info:
        y_offset = text_size[1] + 40
        for info in additional_info:
            cv2.putText(img_with_info, info, (10, y_offset), 
                       font, 0.6, (255, 255, 255), 1)
            y_offset += 25
    
    # 保存图像
    cv2.imwrite(save_path, img_with_info)

def base64_encoder(img):
    img_data = Image.fromarray(img.astype('uint8')).tobytes()
    img_base64 = base64.b64encode(img_data).decode("utf-8")
    return img_base64

def base64_decoder(img_base64, size_out=(128,256)):
    img_data = base64.b64decode(img_base64)
    img = Image.frombytes("RGB", size_out, img_data)
    return np.array(img)

def show_image(img_src, label):
    try:
        ih, iw, _ = img_src.shape
        w = label.geometry().width()
        h = label.geometry().height()
        # keep the original data ratio
        if iw / w > ih / h:
            scal = w / iw
            nw = w
            nh = int(scal * ih)
            img_src_ = cv2.resize(img_src, (nw, nh))
        else:
            scal = h / ih
            nw = int(scal * iw)
            nh = h
            img_src_ = cv2.resize(img_src, (nw, nh))

        frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
        img = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.shape[2] * frame.shape[1],
            QImage.Format_RGB888,
        )
        label.setPixmap(QPixmap.fromImage(img))
    except Exception as e:
        print(traceback.print_exc())

if __name__ == "__main__":
    width, height, channels = 128, 128, 3
    image_array = np.random.randint(0, 256, (width, height, channels), dtype=np.uint8)

    # 将NumPy数组转换为PIL图像
    # image = Image.fromarray(image_array)
    base64_image = base64_encoder(image_array)
    # 将图像转换为Base64字符串
    # buffer = np.array(image)
    # image_data = Image.fromarray(buffer.astype('uint8'))
    # rawBytes = image_data.tobytes()
    # base64_image = base64.b64encode(rawBytes).decode('utf-8')
    img = base64_decoder(base64_image)
    pdb.set_trace()
    # 打印编码后的字符串
    print("Base64 编码后的字符串:")
    print(base64_image)