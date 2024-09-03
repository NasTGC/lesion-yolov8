import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import cv2

model = YOLO("model.pt")

input_folder = "data/test/ct_scans_png"
mask_folder = "data/test/infection_mask_png"


for filename,maskname in zip(os.listdir(input_folder),os.listdir(mask_folder)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)
        mask_path = os.path.join(mask_folder, maskname)

        model.predict(source=img,
              show=True,
              save=False,
              show_labels=True,
              show_conf=True,
              conf=0.2,
              save_txt=False,
              box=False,
              visualize=False
              )
        
    cv2.imshow("mask", cv2.imread(mask_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()