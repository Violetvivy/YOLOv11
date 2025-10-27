from ultralytics import YOLO
import glob
import torch
import gc

model = YOLO('runs/detect/model_based_s2/weights/best.pt')

test_images = glob.glob('E:/Dataset/Fruits 2.v1-three.yolov11/test/images/*.jpg')

results = model.predict(
    source=test_images,
    imgsz=320,      # 输入图像尺寸
    conf=0.40,      # 置信度阈值
    iou=0.45,       # IoU阈值
    device=0,
    save=True,      # 保存预测结果
    save_txt=True,  # 保存标签文件
    save_conf=True, # 保存置信度
    save_crop=True,  # 保存裁剪的检测目标
    batch=4,
    name='based_s2_predictions',  # 结果保存目录名称
)

del results  # 删除引用
torch.cuda.empty_cache()  # 清理
gc.collect()