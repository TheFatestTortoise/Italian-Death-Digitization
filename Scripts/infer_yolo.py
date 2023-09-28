# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:33:40 2023

@author: Mason
"""



from ultralytics import YOLO
from multiprocessing import freeze_support
from ultralytics.yolo.utils.ops import non_max_suppression
import torch
import cv2
iou_thres = 0.5
conf_thres = 0.5
max_det = 100
classes = ['Handwriting']
half = True

model = YOLO(r'F:\LAM\Scripts\runs\detect\train2\weights\best.pt')

if __name__ == '__main__':
    preds = model.predict(r'F:\LAM\Screenshot 2023-09-05 215139.png', conf = 0.09, imgsz = 2560, save_txt = True, max_det = 100)
    img = cv2.imread(r'F:\LAM\Screenshot 2023-09-05 215139.png')
    for box in preds:
        values = box.boxes.xyxy
        for value in values:
            if len(value) == 4:
                print(value)
                x1, y1, x2, y2 = value
                point1 = (int(x1), int(y1))
                point2 = (int(x2), int(y2))
                cv2.rectangle(img, point1, point2, (255, 255, 255), 2)
            else:
                print(len(value))
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()