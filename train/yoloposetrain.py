from ultralytics import YOLO
 
if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'D:\yolo\ultralytics-main\ultralytics-main\yolov8-pose.yaml')  # 也可以使用预训练模型：'yolov8n-pose.pt'
    
    # 训练模型并配置参数
    model.train(
        data=r'D:\yolo\ultralytics-main\ultralytics-main\datayaml\try.yaml',           # 数据配置文件路径
        epochs=10,                # 训练轮次（默认100）
        batch=16,                  # 批量大小（默认16，需根据GPU内存调整）
        imgsz=640,                 # 输入图像尺寸（默认640）
        save=True,                 # 是否保存检查点（默认True）
        resume=True,               # 断点续训
    )