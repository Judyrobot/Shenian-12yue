import sys
sys.path.append("D:\\yolo\\ultralytics-main\\train")
from ultralytics import YOLO
 
if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'D:\yolo\ultralytics-main\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml')  # 也可以使用预训练模型：'yolov8n-pose.pt'

    # 训练模型并配置参数
    model.train(
        data= 'rtdterdata.yaml',           # 数据配置文件路径
        epochs=200,                # 训练轮次（默认100）
        batch=16,                  # 批量大小（默认16，需根据GPU内存调整）
        imgsz=640,                 # 输入图像尺寸（默认640）
        save_period=0,
        workers=0,
        save=True,                 # 是否保存检查点（默认True）
        # resume=True,               # 断点续训
    )