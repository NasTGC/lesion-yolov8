from ultralytics import YOLO
import random
import os
import cv2

model = YOLO("model.pt")

# path = 'path to images'
# mask = 'path to masks'

# sample = random.choice(os.listdir(path))
# mask += sample
# path += sample

model.predict(source='path to image for prediction',
              show=True,
              save=False,
              show_labels=True,
              show_conf=True,
              conf=0.2,
              save_txt=False,
              box=False,
              visualize=False
              )


#cv2.imshow("mask",cv2.imread(mask))
cv2.waitKey(0)
cv2.destroyAllWindows()