from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\Users\User\Documents\GitHub\yolov8_test\runs\detect\generalized_exp3_params\weights\best.pt')
    metrics = model.val(
        data=r'C:\Users\User\Documents\GitHub\yolov8_test\datasets\flying_object_dataset\data.yaml',
        imgsz=416,
        batch=16,
    )
    print(f"전체 mAP50:    {metrics.box.map50:.3f}")
    print(f"전체 mAP50-95: {metrics.box.map:.3f}")