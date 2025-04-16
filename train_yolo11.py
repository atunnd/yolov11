from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="../dataset/data.yaml",  
    epochs=1000,  
    imgsz=640, 
    batch=128, 
    device=0,  
    name="yolov11",
    optimizer="AdamW",
    cos_lr=True,
    lr0=1e-3,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    val=True,
    plots=True,
    exist_ok=True,
    patience=20
)

