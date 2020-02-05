import os
import time

import cv2
import imutils
import numpy as np
import tensorflow as tf


from utils.math_utils import intersect, clear_output
from utils import process_input
from utils import draw_line
from utils import models
from sort import *


# Initialize tracker
tracker = Sort()
memory = {}
entry = 0
exit = 0

args = process_input.parse_user_input()

# initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture(args["input"])
fps = int(vs.get(cv2.CAP_PROP_FPS))
total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
(W, H) = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(args["output"]+"/output.avi", fourcc, fps, (W, H), True)

# get line info
line_a, line_b = draw_line.define_ROI(int(args["line"]), args["input"], H, W)


# load/download model
if args["model"] == "yolov3":
    # load YOLO and COCO class labels
    LABELS, net, ln = models.set_yolo_model()
    # loop over frames from the video file stream
    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

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
        
        try:
            for track in tracks:
                boxes.append([track[0], track[1], track[2], track[3]])
                indexIDs.append(int(track[4]))
                memory[indexIDs[-1]] = boxes[-1]
        except:
            pass

        if len(boxes) > 0:
            i = int(0)

            for box in boxes:
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                cv2.rectangle(frame, (x, y), (w, h), (0,255,0), 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    
                    #cv2.line(frame, p0, p1, color, 3)
                    d_a = ((p0[0] - line_a[0][0])*(line_a[1][1] - line_a[0][1]))- ((p0[1] - line_a[0][1])*(line_a[1][0] - line_a[0][0]))
                    d_b = ((p0[0] - line_b[0][0])*(line_b[1][1] - line_b[0][1]))- ((p0[1] - line_b[0][1])*(line_b[1][0] - line_b[0][0]))

                    if intersect(p0, p1, line_a[0], line_a[1]):
                        print("upper boundary passed")
                        if d_a < 0:
                            entry += 1
                        if d_a > 0:
                            exit += 1

                    if intersect(p0, p1, line_b[0], line_b[1]):
                        print("lower boundary passed")
                        if d_b > 0:
                            entry += 1
                        if d_b < 0:
                            exit += 1

                text = "{}".format(indexIDs[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                i += 1

        print(entry, exit)
        # draw boundary lines
        cv2.line(frame, line_a[0], line_a[1], (0, 255, 255), 3)
        cv2.line(frame, line_b[0], line_b[1], (0, 255, 255), 3)

        # draw counter
        cv2.putText(frame, "Entries: " + str(entry), (50,350), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, "Exits: " + str(exit), (150,350), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)

        # write the output frame to disk
        writer.write(frame)

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()

else: 
    # load TF detection graph and COCO class labels
    detection_graph, category_index = models.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
            # for all the frames that are extracted from input video
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
            
            input_frame = frame

            # expand dimensions to [1, None, None, 3]
            image_np_expanded = np.expand_dims(input_frame, axis=0)

            # detection
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            

            boxes_person = []
            scores_person = []

            scores = np.squeeze(scores).tolist()
            for i in range(np.squeeze(boxes).shape[0]):
                if (scores[i] > 0.5) and (np.squeeze(classes)[i] in category_index.keys()) and (
                    category_index[np.squeeze(classes)[i]]["name"] == "person"):
                    box = tuple(np.squeeze(boxes)[i].tolist())
                    boxes_person.append(box)
                    scores_person.append(scores[i])
 
            boxes = []
            for box in boxes_person:
                (x, y) = (int(box[1]*W), int(box[0]*H))
                (xmax, ymax) = (int(box[3]*W), int(box[2]*H))
                (w, h) = (xmax-x, ymax-y)
                boxes.append([x, y, w, h])

            # non-maxima suppression to suppress weak, overlapping bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, scores_person, args["confidence"], args["threshold"])
            dets = []
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    dets.append([x, y, x+w, y+h, scores_person[i]])
            
            if len(dets) > 0:
                np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
                dets = np.asarray(dets)
                tracks = tracker.update(dets)

            boxes = []
            indexIDs = []
            c = []
            previous = memory.copy()
            memory = {}
            
            try:
                for track in tracks:
                    boxes.append([track[0], track[1], track[2], track[3]])
                    indexIDs.append(int(track[4]))
                    memory[indexIDs[-1]] = boxes[-1]
            except:
                pass 

            if len(boxes) > 0:
                i = int(0)

                for box in boxes:
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))
                    cv2.rectangle(frame, (x, y), (w, h), (0,255,0), 2)

                    if indexIDs[i] in previous:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                        
                        #cv2.line(frame, p0, p1, color, 3)
                        d_a = ((p0[0] - line_a[0][0])*(line_a[1][1] - line_a[0][1]))- ((p0[1] - line_a[0][1])*(line_a[1][0] - line_a[0][0]))
                        d_b = ((p0[0] - line_b[0][0])*(line_b[1][1] - line_b[0][1]))- ((p0[1] - line_b[0][1])*(line_b[1][0] - line_b[0][0]))

                        if intersect(p0, p1, line_a[0], line_a[1]):
                            print("upper boundary passed")
                            if d_a < 0:
                                entry += 1
                            if d_a > 0:
                                exit += 1

                        if intersect(p0, p1, line_b[0], line_b[1]):
                            print("lower boundary passed")
                            if d_b > 0:
                                entry += 1
                            if d_b < 0:
                                exit += 1

                    text = "{}".format(indexIDs[i])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    i += 1

            print(entry, exit)
            # draw boundary lines
            cv2.line(frame, line_a[0], line_a[1], (0, 255, 255), 5)
            cv2.line(frame, line_b[0], line_b[1], (0, 255, 255), 5)

            # draw counter
            cv2.putText(frame, "Entries: " + str(entry), (50,350), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, "Exits: " + str(exit), (150,350), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)

            # write the output frame to disk
            writer.write(frame)

        print("[INFO] cleaning up...")
        writer.release()
        vs.release()

