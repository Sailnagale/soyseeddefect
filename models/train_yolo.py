from ultralytics import YOLO

def initialize_training_pipeline():
    model = YOLO("yolov8n.pt")
    training_arguments = {
        "data": "./config/dataset.yaml",
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "workers": 4,
        "project": "SoySeedDefect_Runs",
        "name": "yolov8n_base_line"
    }
    print("🚀 Initializing deep learning fine-tuning sequence...")
    model.train(**training_arguments)

if __name__ == "__main__":
    initialize_training_pipeline()
