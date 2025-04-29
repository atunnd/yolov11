from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

model.train(
    data="../weather_dataset/weather_dataset/data.yaml", 
    epochs=1000,  
    imgsz=640, 
    batch=64, 
    device=0,  
    name="rtdetr_weather",
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

