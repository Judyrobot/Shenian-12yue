import warnings
 
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    model = YOLO(r"D:\yolo\ultralytics-main\ultralytics-main\runs\pose\train4\weights\best.pt")
    model.val(data=r'\dataset\data_drone1.yaml',
              split='val',
              imgsz=640,
              batch=16,
              project='runs/val',
              name='exp',
              )