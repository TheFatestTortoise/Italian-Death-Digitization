'''
The main body of code:
    1) Check if the image already has a label file created
    2) If it doesn't have a label file, display the image
    3) Tag the image with the following classes:
        
        Switch between the pieces by using w(+), e(-), s(-6) - Switches black and white
    
        0.  White King
        1.  White Queen
        2.  White Rook
        3.  White Bishop
        4.  White Knight
        5.  White Pawn
        6.  Black King
        7.  Black Queen
        8.  Black Rook
        9.  Black Bishop
        10. Black Knight
        11. Black Pawn
        
        Board will be defined by its own button 'b'
        
        12. Board
    in order to tag each of these we will have to wait for an event
    mouse events are handeled by the draw function while keyboard events are handeled
    in the mode function
    
    To switch from a white to black piece +6
    To switch from a black to white piece -6
    
    4) Save the labels to the corresponding label file
        Points must be converted from (class, x1, y1, x2, y2)
        to (class, x_center, y_center, w, h)
    5) Close image
'''
#Imports
import os
import cv2

#Setting up directories
SCRIPT_DIR   = os.getcwd()
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)

#Everything we will be doing in this script should take place within TRAINNING
TRAINNING_DIR = os.path.join(ROOT_DIR, "TRAINING_DATA")

IMAGE_DIR     = os.path.join(ROOT_DIR, 'Raw_image')
LABEL_DIR     = os.path.join(ROOT_DIR, 'Data')



mode = 0
class objectOfInterest():
    def __init__(self, classID, p1, p2):
        self.point1 = p1
        self.point2 = p2
        self.classID = classID
        
def draw(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #If there are no objects or the last object has been filled create a new one
        #This must be done here to not initialize unused objects
        if len(objects) == 0 or objects[-1].point2 != (0, 0):
            objects.append(objectOfInterest(-1, (0, 0), (0, 0)))
            
        if objects[-1].point1 == (0, 0):
            objects[-1].point1 = (x, y)
        else:
            objects[-1].point2 = (x, y)
            print(objects[-1].classID, objects[-1].point1, objects[-1].point2)
            
def draw_rectangle(classID, point1, point2, img):

    colors = [(246, 137, 32), (212, 91, 28), (234, 82, 43), (212, 37, 28), (246, 32, 99), (214, 26, 154), 
              (34, 246, 224), (30, 212, 128), (44, 234, 79), (66, 214, 26), (172, 246, 34), (246, 245, 34),
              (30, 44, 234)]

    cv2.rectangle(img, point1, point2, colors[classID], 2)
    cv2.imshow("RAW DATA", img)
    
def mode_change(key, mode):
    #These two are simple increase and decrease methods
    if key == ord('e'):
        mode = (mode + 1) % 12
    elif key == ord('w'):
        mode = mode - 1
    #This allows the mode switching between black and white
    elif key == ord('s'):
        mode = mode - 6
    #Switch to the board mode
    elif key == ord('d'):
        if mode == 12:
            mode = 0
        else:
            mode = 12
    #Negative overflow loop
    if mode < 0:
        mode = 12 + mode   
    return mode

def export_relative(file, objects, img_w, img_h):
    with open(file, 'w') as w:
        for label in objects:
            x1, y1 = label.point1
            x2, y2 = label.point2
            
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h
            
            w.write("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(label.classID, x_center, y_center, width, height) + "\n")

for image in os.listdir(IMAGE_DIR):
    #Checks if coressponding label exists
    label = os.path.join(LABEL_DIR, image.replace('.png', '.txt'))
    if not os.path.isfile(label):
        #For each image initialize a new list of objects
        objects = []
        drawn_objects = []
        #Read in an load the image
        print(os.path.join(IMAGE_DIR, image))
        img = cv2.imread(os.path.join(IMAGE_DIR, image))
        
        h, w, l = img.shape
        
        cv2.namedWindow("RAW DATA", cv2.WINDOW_AUTOSIZE)
        print(w, h)

        print(w, h)
        img = cv2.resize(img, (w, h))
        
        #Sets mouse events to be handeled by the draw function
        cv2.setMouseCallback("RAW DATA", draw, objects)
        cv2.imshow("RAW DATA", img)
        #Waits for a key to be pressed and assings it to k, waiting for q to be hit to quit
        k = 0
        while(k != ord('q')):
            
            #Drawing the tangle
            for box in objects:
                if box not in drawn_objects and box.point2 != (0, 0):
                    drawn_objects.append(box)
                    draw_rectangle(box.classID, box.point1, box.point2, img)
            
            
            k = cv2.waitKey(1) & 0xFF
            
            if len(objects) != 0:
                mode = mode_change(k, mode)
                objects[-1].classID = mode
        export_relative(label, objects, w, h)
        cv2.destroyAllWindows()
        print(w, h)
        for item in objects:
            print(item.classID, item.point1, item.point2)