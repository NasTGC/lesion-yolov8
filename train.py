from ultralytics import YOLO

model = YOLO('model.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml',
            epochs=5, 
            imgsz=512, 
            batch=64, 
            amp=True, 
            val=True)