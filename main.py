import os
import glob
import argparse
import time

import cv2
import imutils
import numpy as np

from utils import intersect, clear_output
from sort import *

tracker = Sort()
memory = {}
entry = 0
exit = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default = "./input/det_t1_video_00031_test.avi")
ap.add_argument("-o", "--output", help="path to output video", default = "./output/")
ap.add_argument("-y", "--model", help="base path to YOLO directory", default = "yolo")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


YOLO_DIR = "./yolo-obj"

def download_model_weights():
    download_url = "https://pjreddie.com/media/files/yolov3.weights"
    
    print("downloading model weights...")
    opener = urllib.request.URLopener()
    opener.retrieve(download_url, "yolo-coco/yolov3.weights")
    print("model download is complete.")
    return


def load_model():
    model_found = 0
    files = os.listdir(YOLO_DIR)
    
    if "yolov3.weights" in files:
        model_found = 1
    else:
        download_model_weights()

    # load model from path
    print("[INFO] loading YOLO from disk...")
    labelsPath = os.path.sep.join([YOLO_DIR, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = os.path.sep.join([YOLO_DIR, "yolov3.weights"])
    configPath = os.path.sep.join([YOLO_DIR, "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # determine only the output layer names
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return LABELS, net, ln

clear_output()
# load YOLO and COCO class labels
LABELS, net, ln = load_model()

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

# initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

frameIndex = 0

# get the number of frames
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
except:
    print("[INFO] could not determine # of frames in video")
    total = -1

# loop over frames from the video file stream
while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    # get video dimensions
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    # boundary line coordinates
    line = [(0, H//2), (W, H//2)]

    # run YOLO object detector
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize lists of detected bounding boxes, confidences and class IDs
    boxes, confidences, classIDs = [], [], []
    # loop over each of the layer outputs and detections
    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions
            if (confidence > args["confidence"]) and (LABELS[classID] == "person"):
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])
    
    if len(dets) > 0:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)

        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                
                #cv2.line(frame, p0, p1, color, 3)
                d = ((p0[0] - line[0][0])*(line[1][1] - line[0][1]))- ((p0[1] - line[0][1])*(line[1][0] - line[0][0]))
                if intersect(p0, p1, line[0], line[1]):
                    if d < 0:
                        entry += 1
                    if d > 0:
                        exit += 1

            text = "{}".format(indexIDs[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

    print(entry, exit)
    # draw boundary line
    cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

    # draw counter
    cv2.putText(frame, "Entries: " + str(entry), (50,350), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(frame, "Exits: " + str(exit), (150,350), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)

    # saves image file
    cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

    # write to video
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

    if frameIndex >= 4000:
        print("[INFO] cleaning up...")
        writer.release()
        vs.release()
        exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
