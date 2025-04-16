from ultralytics import YOLO
import glob
import pandas as pd
from collections import defaultdict
import cv2

# 加载模型（示例使用 yolov8n.pt ）
model = YOLO('./runs/detect/train_v8_data7/weights/best.pt')
# 定义全局统计字典和单图统计列表
global_counts = defaultdict(int)
per_image_stats = []

# 遍历所有图片（示例路径：images/*.jpg）
for img_path in glob.glob("./datasets/data/11/*.jpg"):
    # 推理单张图片
    results = model.predict(img_path,
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,)
                # exist_ok = True)
    detections = results[0].boxes

    # 提取类别信息并过滤低置信度目标
    conf_threshold = 0.2
    class_ids = detections.cls[detections.conf > conf_threshold].cpu().numpy().astype(int)

    # 统计当前图片的类别
    image_counts = defaultdict(int)
    for cls_id in class_ids:
        cls_name = model.names[cls_id]
        image_counts[cls_name] += 1
        global_counts[cls_name] += 1  # 更新全局统计

    # 记录单图结果
    per_image_stats.append({
        "Image Path": img_path,
        "Total Objects": len(class_ids),
        **image_counts  # 展开类别字典
    })
    # 将统计结果转为 DataFrame
    df_global = pd.DataFrame(list(global_counts.items()), columns=["Class", "Global Count"])
    df_per_image = pd.DataFrame(per_image_stats)

    # 导出 CSV
    df_global.to_csv("global_counts.csv", index=False)
    df_per_image.to_csv("per_image_counts.csv", index=False)