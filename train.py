from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11m.yaml")   # load a pretrained model
    # model = YOLO('runs/detect/type13/weights/best.pt')

    # Train the model
    results = model.train(
        data="E:/Dataset/Fruits15/data.yaml",
        name='E:/VSCode/Yolov11/runs/detect/modelv1.0',
        epochs=100,
        patience=20,
        imgsz=640,
        batch=16,
        lr0=0.001,

        # ===== 学习率调度参数 =====
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        
        # ===== 数据增强参数 =====
        # 颜色增强 - 针对水果颜色特征优化
        hsv_h=0.02,    # 稍微提高色调变化，适应不同成熟度水果
        hsv_s=0.6,     # 降低饱和度变化，保持水果真实颜色
        hsv_v=0.3,     # 降低亮度变化，避免过曝影响识别
        
        # 几何变换 - 保持水果基本形状
        fliplr=0.5,    # 保持水平翻转
        flipud=0.3,    # 降低垂直翻转概率，水果通常有固定方向
        degrees=10.0,  # 旋转角度±10度，增加方向多样性
        translate=0.1, # 平移变换10%，模拟不同拍摄角度
        scale=0.2,     # 缩放20%，适应不同距离拍摄
        
        # 高级增强技术（小数据集调大，防过拟合）
        mosaic=1.0,    # 启用mosaic数据增强
        mixup=0.2,     # 轻微mixup增强，提高泛化能力
        copy_paste=0.1, # 轻微复制粘贴增强
        
        # 其他参数（形状比较固定可减小值）
        shear=1.5,     # 轻微剪切变换
        perspective=0.0005, # 透视变换
        erasing=0.4,   # 随机擦除，模拟遮挡
        
        # 训练优化
        # amp=True,      # 自动混合精度训练
        optimizer="AdamW", # 使用AdamW优化器
        weight_decay=0.0005, # 权重衰减防止过拟合
        dropout=0.1,   # 添加dropout正则化
    )
    # model.train(resume=True)
