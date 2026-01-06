import torch
from ultralytics import YOLO
import os

def main():

    # GPU Check
    print("🔍 Checking GPU availability...")
    if torch.cuda.is_available():
        print("✅ GPU FOUND:", torch.cuda.get_device_name(0))
        device = "0"
    else:
        print("❌ NO GPU FOUND. Using CPU.")
        device = "cpu"

    # Load trained model
    model = YOLO("runs/detect/crime_finetuned/weights/best.pt")

    # Input image
    image_path = "test1.jpg"  # change this to your input image

    results = model.predict(
        source=image_path,
        save=True,
        visualize=True,   # 🔥 enables Grad-CAM heatmaps
        imgsz=640,
        device=device
    )

    print("\n🎉 Inference complete!")
    print("📁 Results saved in: runs/detect/predict/")

    # Extract detected classes
    detections = set()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detections.add(class_name)

    print("\n🧾 Detected Objects:", list(detections))

    return list(detections)


if __name__ == "__main__":
    main()
