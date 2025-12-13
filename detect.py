# from ultralytics import YOLO
 
# yolo = YOLO(r"D:\yolo\ultralytics-main\ultralytics-main\runs\pose\allin\weights\best.pt",task="detect")
# results = yolo(source=r"C:\Users\judy\Desktop\article\信标灯选择实验\shiyanjilushiping\s40a1.2\Snipaste_2022-07-05_16-54-43.png", conf=0.5, device=0, save=True)
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
 
 
if __name__ == '__main__':
    # model = YOLO(r"C:\Users\judy\Desktop\训练好的pt\yolo.pt") # select your model.pt path#原始yolo特征图
    model = YOLO(r"C:\Users\judy\Desktop\新数据集新架构\zuixin数据集\yolo\weights\best.pt") # select your model.pt path#原始yolo特征图

    model.predict(source=r"D:\yolo\ultralytics-main\ultralytics-main\datasets\ULight_new1\images\WangLight",
                  imgsz=640,
                  # project='runs/detect/feature',
                  # name='yolov8',
                  save=True,
                  conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  #visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )
    
    