# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO('./runs/detect/train_v8_3data3/weights/best.pt') # select your model.pt path
#     model.predict(source=r'D:\BaiduNetdiskDownload\codenew\code\datasets\test\images',
#                   imgsz=640,
#                   project='runs/detect',
#                   name='exp',
#                   save=True,
#                 )




import os
import gc
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)


def main():
    try:
        # 1. 初始化模型
        model = YOLO('./runs/detect/96zhong/train_v8_96_1data/weights/best.pt')
        logging.info(" 模型初始化完成")

        # 2. 安全路径处理
        source_dir = r'./datasets/data/test/images'
        source_dir = os.path.normpath(source_dir)
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"路径不存在: {source_dir}")

        # 3. 分批次流式处理,可
        for i, img_file in enumerate(os.listdir(source_dir)):
            img_path = os.path.join(source_dir, img_file)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            # model.predict(source=img_path, imgsz=640, project='runs/detect', name='exp7', save=True)
            model.predict(source=img_path,
                          imgsz=640,
                          project='runs/detect',
                          name='exp12',
                          save=True,
                          exist_ok = True)
            logging.info(f" 已处理第 {i + 1} 张图像: {img_file}")

            if (i + 1) % 100 == 0:  # 每处理100张清理内存
                gc.collect()

    except Exception as e:
        logging.error(f" 进程终止: {str(e)}")
    finally:
        gc.collect()


if __name__ == '__main__':
    main()