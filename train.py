from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO('runs/detect/model_based_s2/weights/last.pt')  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    # results = model.train(data='E:\Dataset\Fruit.v1i.yolov11/data.yaml', epochs=50, imgsz=640, batch=16, device=0, name='model_based_s')
    model.train(resume=True,batch=16)