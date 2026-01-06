from ultralytics import YOLO

model = YOLO("runs/detect/crime_finetuned/weights/best.pt")

model.predict(
    source="dataset/Scene_images/test/images",
    save=True,       
    imgsz=1280,
    augment=True,
    conf=0.19    # TTA = more stable bounding boxes
     
)
