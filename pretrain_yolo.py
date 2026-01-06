import torch
from ultralytics import YOLO

def main():

    print("🔍 Checking GPU availability...")

    if torch.cuda.is_available():
        print("✅ GPU FOUND!")
        print(f"🖥️ GPU Name: {torch.cuda.get_device_name(0)}")
        device = "0"
    else:
        print("❌ GPU NOT FOUND — using CPU")
        device = "cpu"

    model = YOLO("yolov8s.pt")

    model.train(
        data="config/cropped_data.yaml",
        epochs=40,
        imgsz=640,
        batch=16,
        patience=10,
        device=device,
        workers=0,      # IMPORTANT on Windows!!!
        name="crime_pretrain"
    )

    print("\n🎉 Pretraining Complete!")


if __name__ == "__main__":
    main()
