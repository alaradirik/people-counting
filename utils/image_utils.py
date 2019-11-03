import cv2
import numpy as np


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
        print(lines)
        print("Lines are already drawn")

def get_first_frame(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        return image

def draw_lines(video_path):
    global img, img2
    img = get_first_frame(video_path)
    img2 = img.copy()
    cv2.namedWindow('Draw')
    cv2.setMouseCallback('Draw',draw_shape)


    while(True):
        cv2.imshow('Draw',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

draw_lines("../videos/example_02.mp4")
