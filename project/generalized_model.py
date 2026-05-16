from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    model.train(
        data='datasets/flying_object_dataset/data.yaml',
        epochs=163,
        imgsz=416,
        batch=16,
        optimizer='SGD',
        momentum=0.937,
        weight_decay=0.01,
        cls=1.0,
        box=5.5,
        dfl=2.5,
        name='generalized_paper_params'
    )