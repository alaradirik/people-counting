import cv2
import numpy as np


global lines
lines = []

drawing = False
x2,y2 = -1,-1

def draw_shape(event,x,y,flag,parm):
    global x2,y2,drawing, img, img2
    
    if len(lines) < 2:
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
                     1 to manually draw ROI
                     2 to pass in line coordinates
    """
    if line_choice == 0:
        line = [(0, 2*height//5), (width, 2*height//5)]
        return line

    if line_choice == 1:
        lines = draw_lines(video_path)
        print(lines)
        return [lines[0], lines[1]]

    if line_choice == 2:
        user_input = input("Please type in line coordinates as x1 y1 x2 y2: ")
        user_input = user_input.split()
        line = [(int(user_input[0]), int(user_input[1])), (int(user_input[2]), int(user_input[3]))]
        return line


def draw_box(frame, det, color):
    cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), color, 2)
    cv2.putText(frame, 'id = {}'.format(det[4]), (det[0], det[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def annotate_frame(frame, line, entry, exit, H, W):
    # draw boundary lines
    cv2.line(frame, line[0], line[1], (0, 255, 255), 5)
    # draw counter
    cv2.putText(frame, "Entries: " + str(entry), (40, H-40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
    cv2.putText(frame, "Exits: " + str(exit), (240, H-40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
    return frame
