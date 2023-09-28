# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:41:35 2023

@author: Mason
"""

from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
if __name__ == '__main__':
    hyp = dict()
    hyp['lr0']= 0.01 # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    hyp['lrf']= 0.002 # final learning rate (lr0 * lrf)
    hyp['momentum']= 0.937 # SGD momentum/Adam beta1
    hyp['weight_decay']= 0.0005 # optimizer weight decay 5e-4
    hyp['warmup_epochs']= 3.0 # warmup epochs (fractions ok)
    hyp['warmup_momentum']= 0.8 # warmup initial momentum
    hyp['warmup_bias_lr']= 0.1 # warmup initial bias lr
    hyp['box']= 7.5 # box loss gain
    hyp['cls']= 0.5 # cls loss gain (scale with pixels)
    hyp['dfl']= 1.5 # dfl loss gain
    hyp['pose']= 12.0 # pose loss gain
    hyp['kobj']= 1.0 # keypoint obj loss gain
    hyp['label_smoothing']= 0.0 # label smoothing (fraction)
    hyp['nbs']= 64 # nominal batch size
    hyp['hsv_h']= 0.0 # image HSV-Hue augmentation (fraction)
    hyp['hsv_s']= 0.0 # image HSV-Saturation augmentation (fraction)
    hyp['hsv_v']= 0.0 # image HSV-Value augmentation (fraction)
    hyp['degrees']= 10.0 # image rotation (+/- deg)
    hyp['translate']= 0.1 # image translation (+/- fraction)
    hyp['scale']= 0.1 # image scale (+/- gain)
    hyp['shear']= 0.0 # image shear (+/- deg)
    hyp['perspective']= 0.0 # image perspective (+/- fraction), range 0-0.001
    hyp['flipud']= 0 # image flip up-down (probability)
    hyp['fliplr']= 0 # image flip left-right (probability)
    hyp['mosaic']= 0.0 # image mosaic (probability)
    hyp['mixup']= 10.0 # image mixup (probability)
    hyp['copy_paste']= 0.3 # segment copy-paste (probability)
    torch.multiprocessing.freeze_support()


    model = YOLO(r'F:\LAM\Scripts\runs\detect\train12\weights\best.pt')
    
    model.train(data = r'F:\LAM\processsed_death_train.yaml', batch=2, imgsz=2560, translate=0.1, degrees=5.0)