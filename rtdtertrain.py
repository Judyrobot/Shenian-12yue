from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')

model = RTDETR(r"D:\yolo\ultralytics-main\ultralytics-main\rtdter-l-test.yaml")  
#model.load(r'D:\yolo\ultralytics-main\ultralytics-main\runs\detect\train14\weights\last.pt') # 是否加载预训练权重
model.train(data='rtdterdata.yaml',  # 训练参数均可以重新设置
                        epochs=100, 
                        imgsz=640, 
                        workers=0, 
                        batch=16,
                        device=0,
                        optimizer='AdamW',
                        amp=False, 
                        ) 
