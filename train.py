from ultralytics import YOLO

model = YOLO("yolov8n.pt")


# model specification for training
results = model.train(
    data="QR_Dataset/data.yml", 
    epochs=50,
    imgsz=640,
    batch=16, 
    name="qr_yolo_model",        
    project="src/models",       
    augment=True,             
    optimizer='Adam',        
    lr0=0.01,
)

print("✅ Training finished! Model weights saved in src/models")

