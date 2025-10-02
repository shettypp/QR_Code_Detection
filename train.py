# pip install ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # alternatives: yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt


# Train the model
results = model.train(
    data="QR_Dataset/data.yml",  # path to your dataset YAML
    epochs=50,                    # adjust depending on dataset size
    imgsz=640,                    # resize images to 640x640
    batch=16,                     # adjust according to your GPU
    name="qr_yolo_model",         # folder name to save results
    project="src/models",         # output folder for training runs
    augment=True,                 # enable augmentations like flip, mosaic, mixup, HSV
    optimizer='Adam',             # optional, Adam optimizer
    lr0=0.01,                    # initial learning rate (adjust if needed)
)

print("✅ Training finished! Model weights saved in:")
print("src/model/qr_yolo_model/weights/")
