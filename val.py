from ultralytics import YOLO

model = YOLO("G:/Khraya/M2 PFE/image-segmentation-yolov8-main/runs/segment/train5/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
