# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 21:54:45 2023

@author: Mason
"""

import cv2
import numpy as np
import os
import imutils
page_dir = r'F:\LAM\full_pages'

for page in os.listdir(page_dir):
    page_path = os.path.join(page_dir, page)
    if os.path.isfile(page_path):
        scale_factor = 1
        img = cv2.imread(page_path,  cv2.IMREAD_GRAYSCALE)
        img = img
        
        h, w = img.shape
        
        
        #Shrinks image to a viewable size and displays for verification
        
        resized_img = imutils.resize(img, height = 980)
        
        binarized_img = cv2.adaptiveThreshold(resized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 9)
        
        cv2.imshow('image', binarized_img)
        h, w = binarized_img.shape
        print(w, h)
        x = cv2.waitKey(0)
        if x == 32:
        
            cv2.imwrite(r'F:\LAM\Screenshot 2023-09-05 215139.png', binarized_img)
        
        elif x == 27:
            cv2.destroyAllWindows()
            quit()
        cv2.destroyAllWindows()