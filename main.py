from ultralytics import YOLO

model = YOLO("best (4).pt")
model.predict(source="fallenvideo1.mp4", imgsz=640, conf=0.6, save=True)
