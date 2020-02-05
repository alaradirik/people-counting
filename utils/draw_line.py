import cv2
import numpy as np


global lines
lines = []

drawing = False
x2,y2 = -1,-1

def draw_shape(event,x,y,flag,parm):
    global x2,y2,drawing, img, img2
    
    if len(lines) < 4:
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Clicked: ', (x,y))
            lines.append((x, y))
            drawing = True
            img2 = img.copy()
            x2,y2 = x,y
            cv2.line(img,(x2,y2),(x,y),(0,0,255),1, cv2.LINE_AA)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                print('Moving: ',(x,y))
                a, b = x, y
                if a != x & b != y:
                    img = img2.copy()
                    cv2.line(img,(x2,y2),(x,y),(0,255,0),1, cv2.LINE_AA)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            print('Released: ',(x,y))
            lines.append((x, y))
            img = img2.copy()
            cv2.line(img,(x2,y2),(x,y),(0,0,255),1, cv2.LINE_AA)
    else:
        return

def get_first_frame(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        print("yes")
        return image

def draw_lines(video_path):
    global img, img2
    img = get_first_frame(video_path)
    img2 = img.copy()
    cv2.namedWindow("Draw")
    cv2.setMouseCallback("Draw",draw_shape)
    
    # press the escape button to exit
    while True:
        cv2.imshow("Draw",img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
  
    return lines

def define_ROI(line_choice, video_path, height, width):
    """
    Define Region of Interest based on user input.
    Input Arguments:
        line_choice: 0 for automatic ROI assignment
                     1 to define ROI bounded by two lines
    """
    if line_choice == 0:
        line_a = [(0, height//5), (width, height//5)]
        line_b = [(0, 3*height//5), (width, 3*height//5)]
        return line_a, line_b

    if line_choice == 1:
        lines = draw_lines(video_path)
        return [lines[0], lines[1]], [lines[2], lines[3]]
