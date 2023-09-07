# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 21:54:45 2023

@author: Mason
"""

import cv2
import numpy as np
import os

scale_factor = 1
img = cv2.imread(r'F:\LAM\full_pages\Screenshot 2023-09-05 215139.png',  cv2.IMREAD_GRAYSCALE)
img = img

h, w = img.shape


#Shrinks image to a viewable size and displays for verification
viewable_width = w // scale_factor
viewable_height = h // scale_factor


resized_img = cv2.resize(img, (viewable_width, viewable_height))

ret, binarized_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.array([[0, -1, 0],
                   [-1, 20,-1],
                   [0, -1, 0]])

sharp = cv2.filter2D(binarized_img, -1, kernel)
sharp_boxes = sharp.copy()
adjusted_lines = []

cv2.imshow('image', sharp_boxes)
print(adjusted_lines)
h, w = sharp.shape
print(w, h)
x = cv2.waitKey(0)
if x == 32:

    cv2.imwrite(r'F:\LAM\Screenshot 2023-09-05 215139.png', sharp)

elif x == 27:
    cv2.destroyAllWindows()
    quit()
cv2.destroyAllWindows()