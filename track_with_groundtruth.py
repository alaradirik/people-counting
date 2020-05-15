import os
import time

import cv2
import imutils
import dlib
import numpy as np

from utils import get_input_args, math_utils, file_utils
from utils import image_utils, model_utils

from trackers.sort import Sort
tracker = Sort(use_dlib= True)

# Initialize tracker
entry = 0
exit = 0

args = get_input_args.parse_user_input()
file_utils.create_paths()

# initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture("./input_videos/TownCentreXVID.mp4")
fps = int(vs.get(cv2.CAP_PROP_FPS))
total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
(W, H) = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("output/output.mp4", fourcc, fps, (W, H), True)

line = image_utils.define_ROI(int(args["line"]), "./input_videos/TownCentreXVID.mp4", H, W)
detector = model_utils.GroundTruthDetections()

frame_index = 0
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    detections = detector.get_detected_items(frame_index)
    trackers = tracker.update(detections, frame)

    current = {}
    for d in trackers:
        d = d.astype(np.int32)

        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0,255,0), 2)
        cv2.putText(frame, 'id = {}'.format(d[4]), (d[0], d[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if detections != []:
            cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        current[d[4]] = (d[0], d[1], d[2], d[3])
        if d[4] in tracker.previous:
            previous_box = tracker.previous[d[4]]
            entry, exit = math_utils.compare_with_prev_position(previous_box, d, line, entry, exit)

    tracker.previous = current

    frame = image_utils.annotate_frame(frame, line, entry, exit, H, W)
    
    writer.write(frame)
    frame_index += 1


vs.release()
