from ultralytics import YOLO

if __name__ == '__main__':
    experiments = [
        # 논문 최종값 (기준)
        {'box': 5.5, 'cls': 1.0, 'dfl': 2.5, 'weight_decay': 0.01},
        # box loss 변형
        {'box': 7.5, 'cls': 1.0, 'dfl': 2.5, 'weight_decay': 0.01},
        # cls loss 변형
        {'box': 5.5, 'cls': 0.5, 'dfl': 2.5, 'weight_decay': 0.01},
        # weight decay 변형
        {'box': 5.5, 'cls': 1.0, 'dfl': 2.5, 'weight_decay': 0.0005},
    ]

    for i, params in enumerate(experiments):
        model = YOLO('yolov8m.pt')
        model.train(
            data='datasets/flying_object_dataset/data.yaml',
            epochs=10,
            imgsz=416,
            batch=16,
            optimizer='SGD',
            momentum=0.937,
            name=f'hyperparam_exp_{i}',
            **params
        )