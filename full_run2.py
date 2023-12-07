# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:37:04 2023

@author: ianva
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:58:44 2023

@author: Mason
"""
import os
from ultralytics import YOLO
import cv2
import imutils
import numpy as np
import copy
#model = YOLO(r'F:\LAM\Scripts\runs\detect\train2\weights\best.pt')
cwd = os.getcwd();
default_size = 600
imagedir = os.path.join(cwd, "Images")
model_dir = os.path.join(cwd, "best.pt")
header_height = 40
scale_factor = 2560/980
point_map = {
    "day": (0.898792, 0.0716309),
    
    "residence": (0.47591, 0.257581),
    "month": (0.283924, 0.260245 ),
    "name": (0.319623, 0.10419)
}
important_fields = point_map.keys()
class ROI:
    def __init__(self, pt1, pt2, classNUM):
        self.classID = classNUM
        self.pt1 = pt1
        self.pt2 = pt2
        self.calculate_properties()
    def print_info(self):
        self.calculate_properties()
        print('\n ________',
              '\n|  Class |', self.classID,
              '\n| Point1 |', self.pt1,
              '\n| Point2 |', self.pt2,
              '\n|   W    |', self.w,
              '\n|   H    |', self.h,
              '\n|  X_C   |', self.x_center,
              '\n|  Y_C   |', self.y_center,
              '\n ¯¯¯¯¯¯¯¯')
    def calculate_properties(self):
        try:
            self.w = abs(self._x2 - self._x1)
            self.h = abs(self._y2 - self._y1)
            self.x_center = int((self._x2 + self._x1)//2)
            self.y_center = int((self._y2 + self._y1)//2)
        except:
            print('Property calculation unsucessful, missing value')
    @property
    def x1(self):
        return(self._x1)
    @x1.setter
    def x1(self, x):
        try:
            if x < self.x2:
                self._x1 = x
            else:
                self._x2 = x
        except:
            self._x1 = x
    @property
    def x2(self):
        return(self._x2)
    @x2.setter
    def x2(self, x):
        try:
            if x > self.x1:
                self._x2 = x
            else:
                self._x1 = x
        except:
            self._x2 = x
    @property
    def y1(self):
        return(self._y1)
    @y1.setter
    def y1(self, y):
        try:
            if y < self.y2:
                self._y1 = y
            else:
                self._y2 = y
        except:
            self._y1 = y
    @property
    def y2(self):
        return(self._y2)
    @y2.setter
    def y2(self, y):
        try:
            if y > self.y1:
                self._y2 = y
            else:
                self._y1 = y
        except:
            self._y2 = y

    @property
    def pt1(self):
        return(self._x1, self._y1)
    @pt1.setter
    def pt1(self, pt1):
        self._x1, self._y1 = pt1
    
    @property
    def pt2(self):
        return(self._x2, self._y2)
    @pt2.setter
    def pt2(self, pt2):
        self._x2, self._y2 = pt2

    def roboflow_form(self):
        self.calculate_properties()
        return(self.classID, self.x_center, self.y_center, self.w, self.h)
    
    def from_roboflow_form(self, classID, x_center, y_center, w, h):

        self.classID = classID
        self._x1 = int(x_center - (w//2))
        self._y1 = int(y_center - (h//2))
        self._x2 = int(x_center + (w//2))
        self._y2 = int(y_center + (h//2))
        self.calculate_properties()

def reduceAll(boxes):
    merged_boxes = []
    count = 0
    for roi in boxes:
        # Check if the current ROI overlaps with any of the merged boxes
        overlap_found = False
        for i in range(len(merged_boxes)):
            merged_box = merged_boxes[i]
            if merged_box.x1 < roi.x_center < merged_box.x2 and merged_box.y1 < roi.y_center < merged_box.y2:
                merged_boxes[i].x1 = min(roi.x1, merged_box.x1)
                merged_boxes[i].y1 = min(roi.y1, merged_box.y1)
                merged_boxes[i].x2 = max(roi.x2, merged_box.x2)
                merged_boxes[i].y2 = max(roi.y2, merged_box.y2)
                overlap_found = True

        if not overlap_found:
            # If no overlapping merged box is found, add the ROI as a new merged box
            merged_boxes.append(roi)
        count += 1
    return merged_boxes
 
def find_important_boxes(point_map, boxes, img_shape, R = 500):
    output_dict = {}
    
    # Iterate through the boxes and check if they are within the radius of any point
    for box in boxes:
        for point_name, point_coords in point_map.items():
            box.calculate_properties()
            if (abs((point_coords[0] * img_shape[0]) - box.x_center) <= R and abs((point_coords[1] * img_shape[1]) - box.y_center) <= R):
                # The box is within the radius of this point
                output_dict[point_name] = box
                found = True
    print(output_dict)
    for name, ROI in output_dict.items(): ROI.print_info()
    return output_dict

def create_spreadsheet(region_dict, important_fields, img, axis_s):
    first_field = True
    white_box = np.ones((default_size, default_size), np.uint8) * 255
    white_box.reshape((default_size, default_size))
    
    
    for field in important_fields:
        box = region_dict.get(field, "")
        if box != "":
            
            crop = img[box.y1:box.y2, box.x1:box.x2]
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if axis_s:
               crop = imutils.resize(crop, height = default_size)
            else:
               crop = imutils.resize(crop, width = default_size)
            
            if first_field:
                out_img = crop
                if out_img.shape[0] >= 700:
                    out_img = out_img[0:700,:]
                if out_img.shape[1] >= 9000:
                    out_img = out_img[:,0:9000]
                    
                horz_spacer = np.zeros((out_img.shape[0],10000-out_img.shape[1]), np.uint8)
                out_img = np.concatenate((out_img,horz_spacer), axis = axis_s)
                vert_spacer = np.zeros((800-out_img.shape[0], out_img.shape[1]), np.uint8)
                out_img = np.concatenate((out_img,vert_spacer), axis = 0)
                first_field = False
                                        
            else:
                if crop.shape[0] >= 700:
                    crop = crop[0:700,:]
                if crop.shape[1] >= 9000:
                    crop = crop[:,0:9000]
                horz_spacer = np.zeros((crop.shape[0],10000 - crop.shape[1]), np.uint8)
                a = np.concatenate((crop,horz_spacer), axis=axis_s)
                vert_spacer = np.zeros((800-a.shape[0],a.shape[1]), np.uint8)
                b = np.concatenate((a,vert_spacer), axis=0)
                out_img = np.concatenate((out_img,b), axis = axis_s)

                '''
                cv2.imshow("test", out_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
        else:
            if first_field:
                out_img = white_box
                first_field = False
                                        
            else:
                out_img = np.concatenate((out_img, white_box), axis=axis_s)
                horz_spacer = np.zeros((out_img.shape[0],100), np.uint8)
                out_img = np.concatenate((out_img,horz_spacer), axis = axis_s)
                
            

    return out_img
    
if __name__ == '__main__':
    boxes = []
    first_file = True
    if len(os.listdir(imagedir)) > 1:
        conc_axis = 1
    else:
        conc_axis = 0
    for file in os.listdir(imagedir):
        if file != "temp.png":

            print(file)
            preds = []
            filepath = os.path.join(imagedir, file)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = imutils.resize(img, width = 2560)
            img_h, img_w = img.shape
            '''
            Reading in predictions without inference comment out this block an uncomment
            lower block for inference
            '''
            '''
            with open(, 'r') as f:
                lines = f.readlines()
                imgRect = copy.deepcopy(img)
                for line in lines:
                    line = line.replace('\n', '').split(' ')
                    box = ROI((0, 0), (0, 0), 0)
                    #print(line)
                    #print(int(float(line[1]) * img_w))
                    #print(img_w)
                    box.from_roboflow_form(int(line[0]), int(float(line[1]) * img_w), int(float(line[2]) * img_h),
                                      int(float(line[3]) * img_w), int(float(line[4]) * img_h))
                    imgRect = cv2.rectangle(imgRect, (box.x1, box.y1), (box.x2, box.y2), (255, 255, 255), thickness = 3)
                    
                    #box.print_info()
                    preds.append(box)
            print(preds)
            '''
            model = YOLO(model_dir)
            bw_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
            cv2.imwrite(os.path.join(imagedir, 'temp.png'), bw_img)
            bw_img = cv2.imread(os.path.join(imagedir, 'temp.png'))
            boxes = model.predict(bw_img, conf = 0.05, imgsz = 2560, save_txt = True, save = True, max_det = 170)
            for box in boxes:
                values = box.boxes.xyxy
                for value in values:
                    if len(value) == 4:
                        x1, y1, x2, y2 = value
                        a =ROI((int(x1), int(y1)),(int(x2), int(y2)),0)
                        preds.append(a)
                
            
            
            
            reduced_boxes = reduceAll(preds)
            region_dict = find_important_boxes(point_map, reduced_boxes, (img_w, img_h))
    
            
            if first_file:
                spreadsheet = create_spreadsheet(region_dict, important_fields, img, conc_axis)
                spreadsheet = imutils.resize(spreadsheet, width = 600)
    
                first_file = False
            else:
                new_sheet = create_spreadsheet(region_dict, important_fields, img, conc_axis)
                new_sheet = imutils.resize(new_sheet, width = 600)
                spreadsheet = np.concatenate((spreadsheet, new_sheet), 0)
        
    spreadsheet_size = spreadsheet.shape
    
    
    header_block= np.ones((header_height, spreadsheet_size[1]), np.uint8) * 255
    file_block = np.ones((spreadsheet_size[0] + header_height, 100), np.uint8) * 255

    spreadsheet = np.concatenate((header_block, spreadsheet), 0)
    spreadsheet = np.concatenate((file_block, spreadsheet), 1)
    
    spreadsheet = cv2.putText(spreadsheet, 'day           month           city           name', (120,20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 1, cv2.LINE_AA)
    y = 45;
    for name in os.listdir(imagedir):
        if name != "temp.png":
            spreadsheet = cv2.putText(spreadsheet, name[0:6], (10,y), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 0), 1, cv2.LINE_AA)
            y = y+12;
    spreadsheet = cv2.resize(spreadsheet, (0,0),fx=2,fy=2)
    
    cv2.imshow("img", spreadsheet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
