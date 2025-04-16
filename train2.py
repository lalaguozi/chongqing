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
    workers = 2
    batch = 12 # 注意：如果增大 imgsz，可能需要减小 batch size

    data_name = "data5"
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

    # # 指定 hyp_pest.yaml 的路径 (如果需要覆盖默认超参数，可以取消注释并使用 hyp)
    # hyp_path = abs_path('datasets/data/hyp_pest.yaml', path_type='current')
    # unix_style_hyp_path = hyp_path.replace(os.sep, '/')

    # 选择加载模型的方式
    model = YOLO(model='./ultralytics/cfg/models/v12/yolov12l.yaml', task='detect')  # 从 yaml 构建新模型
    # model = YOLO('runs/detect/train_v12_haichongdata5/weights/last.pt')  # 加载上次训练的 last.pt 继续训练
    # model = YOLO(model='./weights/yolov12l.pt') # 加载预训练权重开始新训练

    # --- 训练参数设置 ---
    current_imgsz = 800 # 你当前的设置
    # 尝试增大 imgsz (如果 GPU 显存允许)
    # current_imgsz = 1024 # 例如，尝试 1024
    # current_imgsz = 1280 # 或者更大

    print(f"Starting training with imgsz={current_imgsz}, batch={batch}")

    results2 = model.train(
        data=data_path,             # 数据配置文件路径
        device=device,              # 设备
        # resume=True,              # 如果要严格从 last.pt 的状态恢复（包括优化器状态、epoch数），则设置 resume=True
        workers=workers,            # 数据加载的工作进程数
        imgsz=current_imgsz,        # **图像大小** - 尝试增大 (800 已经比默认 640 大)
        epochs=200,                 # 训练轮数
        batch=batch,                # **批次大小** - 如果增大 imgsz 导致 OOM，需要减小此值
        # name='train_v12_haichong' + data_name + f'_imgsz{current_imgsz}', # 训练任务名称，加入尺寸信息更好区分
        name='train_v12_haichong' + data_name,

        # --- 关键参数：增强尺度鲁棒性 ---
        # multi_scale=True,           # **启用多尺度训练** - 在 imgsz +/- 50% 范围内随机调整输入尺寸
        # hyp=unix_style_hyp_path, # 如果你想用自定义的超参数文件覆盖下面的参数，取消注释这行

        # --- 确认或调整数据增强参数 ---
        # 这些参数直接在 train() 中设置会覆盖默认或 hyp 文件中的设置
        scale=0.9,                  # **图像缩放增强因子** - 0.9 表示缩放范围为 [0.1, 1.9]*imgsz，已经是很大的范围了，有利于尺度学习。保持或按需微调。
        mosaic=1.0,                 # **Mosaic 增强** - 概率 1.0 表示始终启用。强烈推荐，极大增强尺度和上下文多样性。
        mixup=0.15,                 # **Mixup 增强** - 概率 0.15。有助于模型泛化。
        copy_paste=0.5,             # **Copy-paste 增强** - 概率 0.5。非常有用于增加目标在不同背景和尺度的出现次数，尤其适合你的场景（模拟大目标）。可以考虑适当增加，如 0.6 或 0.7，如果数据足够。

        # --- 可以考虑取消注释以启用其他标准增强 ---
        # hsv_h=0.015,  # 色调 (默认值或根据 hyp 文件)
        # hsv_s=0.7,    # 饱和度
        # hsv_v=0.4,    # 亮度
        degrees=10.0,   # **旋转角度** (+/- 10 度) - 轻微增加有助于鲁棒性
        translate=0.1,  # **平移** (图像尺寸的 +/- 10%)
        shear=0.0,      # 剪切 (通常保持 0 或很小的值)
        perspective=0.0,# 透视变换 (通常保持 0 或很小的值)
        flipud=0.1,     # **上下翻转概率** - 如果你的害虫存在上下颠倒的情况，可以适当增加
        fliplr=0.5,     # **左右翻转概率** - (默认值 0.5) 通常保持

        # --- 其他训练参数 ---
        optimizer='AdamW',          # 可以尝试 AdamW 优化器，有时效果更好 (默认可能是 SGD 或 AdamW，取决于版本)
        # lr0=0.002,                # 初始学习率 (如果更改优化器或模型，可能需要调整)
        # lrf=0.01,                 # 最终学习率因子 (lr0 * lrf)
        patience=20,                # **提前停止的耐心轮数** - 可以适当增加到 30 或 50，给模型更多时间学习
        # freeze = 10,              # 冻结骨干网络层数 (仅在迁移学习微调时有用)
        # weight_decay=0.0005,      # 权重衰减
    )

  