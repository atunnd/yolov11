from ultralytics import YOLO

def run():
    model = YOLO("yolo11n.pt")
    train_results = model.train(
        data = "../dataset/data.yaml",
        epochs=1000,
        batch=128,
        device=0,
        plots=True
    )

if __name__ == "__main__":
    run()