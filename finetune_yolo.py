import torch
from ultralytics import YOLO

# ------------------------------------------------------
# GPU CHECK
# ------------------------------------------------------
print("🔍 Checking GPU availability...")

if torch.cuda.is_available():
    print("✅ GPU FOUND!")
    print(f"🖥️ GPU Name: {torch.cuda.get_device_name(0)}")
    device = "0"
else:
    print("❌ GPU NOT FOUND — using CPU")
    device = "cpu"

# ------------------------------------------------------
# LOAD PRETRAINED WEIGHTS
# ------------------------------------------------------
model = YOLO("runs/detect/crime_pretrain/weights/best.pt")

# ------------------------------------------------------
# FINETUNE ON SCENE IMAGES
# ------------------------------------------------------
model.train(
    data="config/scene_data.yaml",
    epochs=60,
    imgsz=640,
    batch=8,
    lr0=0.0005,
    patience=15,
    device=device,         # <--- GPU enabled
    workers=0,
    name="crime_finetuned"
)

print("\n🎉 Finetuning Completed!")
print("🔥 Model saved inside: runs/detect/crime_finetuned/")
