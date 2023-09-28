# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:39:01 2023

@author: Mason
"""

from bs4 import BeautifulSoup
import os
import cv2
import numpy as np
scale_factor = 3

root_dir = os.path.dirname(os.getcwd())
page_dir = os.path.join(root_dir, 'full_pages')
data_dir = os.path.join(root_dir, 'processed_data')
img_dir = os.path.join(root_dir, 'processed_img')

def pixel_to_relative(pixel, img_shape, output_file = ''):
    #pixel = [[x1, y1, x2, y2, classID]]
    #img_shape = (w, h)
    w, h = img_shape
    output = []
    written = False
    
    print(w, h)
    
    for box in pixel:
        x1_pix, y1_pix, x2_pix, y2_pix, classID = box
        
        x1_rel = x1_pix / w
        x2_rel = x2_pix / w
        y1_rel = y1_pix / h
        y2_rel = y2_pix / h
        
        print(x1_rel, x2_rel, y1_rel, y2_rel)
        print('')
        
        width = abs(x1_rel - x2_rel)
        height = abs(y1_rel - y2_rel)
        
        x_center = min(x1_rel, x2_rel) +(width/2)
        y_center = min(y1_rel, y2_rel) + (height/2)
        
        output.append([x_center, y_center, width, height, classID])
        if output_file != '':
            with open(output_file, 'a') as f:
                if not written:
                    f.write("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(classID, x_center, y_center, width, height))
                    written = True
                else:
                    f.write("\n{} {:.3f} {:.3f} {:.3f} {:.3f}".format(classID, x_center, y_center, width, height))
    if len(pixel) == 0:
        with open(output_file, 'w') as f:
            print('No data')
    return output
        
for directory in os.listdir(page_dir):
    dir_name = os.path.join(page_dir, directory)
    images = os.path.join(dir_name, 'img')
    xmls = os.path.join(dir_name, 'xml')
    for image, xml in zip(os.listdir(images), os.listdir(xmls)):
        data_name = xml.replace('.xml', '.txt')
        data_path = os.path.join(data_dir, data_name)
        if image.replace('.jpg', '.xml') == xml and not os.path.isfile(data_path) :
            print(image)
            
            #Reads in xml file to readable data
            with open(os.path.join(xmls, xml), 'r') as f:
                data = f.read()
            
            bs_data = BeautifulSoup(data, 'xml')
            lines = bs_data.find_all('line', rotation='0.0')  
            img = cv2.imread(os.path.join(images, image))
            
            #For each box, finds coodinates and draws them on the image

            x_bottom = 10000
            y_bottom = 10000
            x_top = 0
            y_top = 0
            formatted_lines = []
            for line in lines:
                x1 = int(float(line['x']))
                y1 = int(float(line['y']))
                w = int(float(line['w']))
                h = int(float(line['h']))
                x2 = x1 + (w)
                y2 = y1 + (h)

                x_bottom = min(x1, x2, x_bottom)
                y_bottom = min(y1, y2, y_bottom)
                
                x_top = max(x1, x2, x_top)
                y_top = max(y1, y2, y_top)
                
                formatted_lines.append([x1, y1, x2, y2, 0])
            
                
                
            w_og, h_og, l_og = img.shape
            
            
            if x_top == 0:
                y_top = h_og
                x_top = w_og
                y_bottom = 0
                x_bottom = 0

            
            cropped_img = img[y_bottom:y_top, x_bottom:x_top, 2]
            h, w= cropped_img.shape
            #Shrinks image to a viewable size and displays for verification
            viewable_width = w // scale_factor
            viewable_height = h // scale_factor

            
            resized_img = cv2.resize(cropped_img, (viewable_width, viewable_height))
            
            ret, binarized_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            '''
            kernel = np.array([[0, -1, 0],
                               [-1, 20,-1],
                               [0, -1, 0]])
            
            sharp = cv2.filter2D(binarized_img, -1, kernel)
            sharp_boxes = sharp.copy()'''
            adjusted_lines = []
            for box in formatted_lines:
                x1, y1, x2, y2, classID = box
                
                x1_adjusted = (x1 - x_bottom) // scale_factor
                x2_adjusted = (x2 - x_bottom) // scale_factor
                y1_adjusted = (y1 - y_bottom) // scale_factor
                y2_adjusted = (y2 - y_bottom) // scale_factor
                
                adjusted_lines.append([x1_adjusted, y1_adjusted, x2_adjusted, y2_adjusted, 0])
                
                cv2.rectangle(binarized_img, (x1_adjusted, y1_adjusted), (x2_adjusted, y2_adjusted), (255, 255, 255), 2)
                
            cv2.imshow('image', binarized_img)
            print(adjusted_lines)
            h, w = binarized_img.shape
            print(w, h)
            x = cv2.waitKey(0)
            if x == 32:
                
                pixel_to_relative(adjusted_lines, (w, h), data_path)
                cv2.imwrite(os.path.join(img_dir, image), binarized_img)
            elif x == ord('q'):
                continue
            elif x == 27:
                cv2.destroyAllWindows()
                quit()
            cv2.destroyAllWindows()
            
            
            