from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11m.yaml")  # build a new model from YAML
    model = YOLO('runs/detect/model_based_m2/weights/last.pt')  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    '''
    results = model.train(
        data='roboflow/data.yaml', 
        name='model_based_m2',
        epochs=100, 
        # patience=20, 
        imgsz=640, 
        batch=32, 
        device=0, 
        # 数据增强参数
        hsv_h=0.02,  # 图像色调增强
        hsv_s=0.6,    # 图像饱和度增强  
        hsv_v=0.3,    # 图像亮度增强

        # 几何变换
        fliplr=0.5,    # 保持水平翻转
        flipud=0.3,    # 垂直翻转概率
        degrees=10.0,  # 旋转角度±10度，增加方向多样性
        translate=0.1, # 平移变换10%，模拟不同拍摄角度
        scale=0.2,     # 缩放20%，适应不同距离拍摄
        
        # 高级增强
        mosaic=1.0,    # 启用mosaic数据增强
        mixup=0.1,     # 轻微mixup增强，提高泛化能力
        copy_paste=0.1, # 轻微复制粘贴增强
        
        # 其他
        shear=2.0,     # 轻微剪切变换
        perspective=0.0005, # 透视变换
        erasing=0.4,   # 随机擦除，模拟遮挡
    )
    '''
    model.train(resume=True)