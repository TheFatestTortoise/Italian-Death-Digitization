# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:41:35 2023

@author: Mason
"""

from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()


    model = YOLO(r'C:\Users\mrbrady1\Desktop\ml\Scripts\runs\detect\train12\weights\best.pt')
    
    model.train(data = r'C:\Users\mrbrady1\Desktop\ml\processed_train.yaml', batch=2, imgsz = 1280)