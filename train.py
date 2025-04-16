import os
import torch
import yaml
from ultralytics import YOLO  # 导入YOLO模型
from QtFusion.path import abs_path
import matplotlib.pyplot as plt
import matplotlib as mpl # 导入 matplotlib 库


# 在进行任何绘图之前设置
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Noto Sans CJK SC'] # 指定优先使用的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

device = "0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

if __name__ == '__main__':  # 确保该模块被直接运行时才执行以下代码
    workers =2
    batch = 12

    data_name = "data6"
    data_path = abs_path(f'datasets/{data_name}/data.yaml', path_type='current')  # 数据集的yaml的绝对路径
    unix_style_path = data_path.replace(os.sep, '/')

    # 获取目录路径
    directory_path = os.path.dirname(unix_style_path)
    # 读取YAML文件，保持原有顺序
    with open(data_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    # 修改path项
    if 'path' in data:
        data['path'] = directory_path
        # 将修改后的数据写回YAML文件
        with open(data_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)

    # # 指定 hyp_pest.yaml 的路径
    # hyp_path = abs_path('datasets/data/hyp_pest.yaml', path_type='current')  # 使用 abs_path 获取绝对路径
    # unix_style_hyp_path = hyp_path.replace(os.sep, '/')

    # model = YOLO(model='./ultralytics/cfg/models/v12/yolov12l.yaml', task='detect')  #改动模型
    model = YOLO('runs/detect/train_v12_haichongdata5/weights/best.pt')  # 加载检查点
    # model = YOLO(model='./weights/yolov12l.pt') #使用预训练模型
    results2 = model.train(
        data=data_path,  # 数据配置文件路径
        device=device,  # 设备（例如 'cuda' 或 'cpu'）
        # resume=True, 
        workers=workers,  # 数据加载的工作进程数
        imgsz=800,  # 图像大小，640、768
        epochs=600,  # 训练轮数l00
        batch=batch,  # 批次大小 
        name='train_v12_haichong' + data_name,  # 训练任务名称
        # optimizer='SGD',#使用手动设置参数
        # lrf=0.1,  # 原0.01 → 提升至初始值的10%
        # cos_lr = True,  # 启用余弦退火调度
        # lr0=0.002,  # 初始学习率从0.01降低（适配预训练权重）
        # momentum=0.937,  # 动量
        # weight_decay=0.0005,  # 权重衰减
        # warmup_epochs=3.0,  # 预热轮数
        # box = 5, # 边界框损失权重（可适当增加）
        # cls = 2,  # 分类损失权重
        # dfl = 1.5, # 分布焦点损失权重
        # hsv_h=0.05,  # HSV 色调增强
        # hsv_s=0.8,  # HSV 饱和度增强
        # hsv_v=0.5,  # HSV 亮度增强
        # degrees=25.0,  # 旋转角度
        # translate=0.2,  # 平移
        # scale=0.7,  # 缩放
        # shear=0.0,  # 剪切
        # perspective=0.1,  # 透视变换
        # flipud=0.2,  # 上下翻转概率
        # fliplr=0.5,  # 左右翻转概率
        # mosaic=1.0,  # 马赛克增强
        # mixup=0.7,  # Mixup 增强
        # copy_paste=0.3,  # Copy-paste 增强
        # conf = 0.25,
        # iou = 0.5,
        # patience =20,  # 提前停止
        # # freeze = 10,  # 冻结前 10 层
        scale=0.9,  # S:0.9; M:0.9; L:0.9; X:0.9
        mosaic=1.0,
        mixup=0.15,  # S:0.05; M:0.15; L:0.15; X:0.2
        copy_paste=0.6,  # S:0.15; M:0.4; L:0.5; X:0.6
    )

    # from ultralytics import YOLO
    #
    # model = YOLO('yolov12n.yaml')
    #
    # # Train the model
    # results = model.train(
    #     data='coco.yaml',
    #     epochs=600,
    #     batch=256,
    #     imgsz=640,
    #     scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
    #     mosaic=1.0,
    #     mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
    #     copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
    #     device="0,1,2,3",
    # )
    #
    # # Evaluate model performance on the validation set
    # metrics = model.val()
    #
    # # Perform object detection on an image
    # results = model("path/to/image.jpg")
    # results[0].show()
