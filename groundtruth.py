
import os
import time
import json
import random

import cv2
import imutils
import dlib
import numpy as np

from utils import math_utils
from utils import image_utils

# Initialize tracker
entry = 0
exit = 0

# initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture("./input_videos/TownCentreXVID.mp4")
fps = int(vs.get(cv2.CAP_PROP_FPS))
total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
(W, H) = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("output/output.mp4", fourcc, fps, (W, H), True)

# get line info
line = [(661, 166), (1859, 520)]

with open('./ground_truth/data.txt') as json_file:
    data = json.load(json_file)


detections = np.loadtxt('./ground_truth/TownCentre-groundtruth.top', delimiter=',')
colours = [(random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256)) for i in range(len(list(set(detections[:, 0].tolist()))))]

# for all the frames that are extracted from input video
frame_index = 0

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    frame_dets = detections[detections[:, 1] == frame_index, :]

    for d in frame_dets:
        d = d.astype(np.int32)
        cv2.rectangle(frame, (d[8], d[9]), (d[10], d[11]), colours[d[0]], 2)
        cv2.putText(frame, 'id = {}'.format(d[0]), (d[8], d[9] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colours[d[0]], 2)
        
    entry += data['counts']['frame_{}'.format(frame_index)]['entries']
    exit += data['counts']['frame_{}'.format(frame_index)]['exits']
    frame = image_utils.annotate_frame(frame, line, entry, exit, H, W)
    
    writer.write(frame)
    frame_index += 1

writer.release()
vs.release()
