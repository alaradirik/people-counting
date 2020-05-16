import os
import time
import random

import cv2
import imutils
import dlib
import numpy as np
import tensorflow as tf

from utils import get_input_args, math_utils, file_utils
from utils import image_utils, model_utils

from trackers.deep_sort import preprocessing, nn_matching
from trackers.deep_sort.detection import Detection
from trackers.deep_sort.tracker import Tracker
from utils import generate_detections as gdet

# Initialize tracker
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort 
model_filename = './trackers/deep_sort/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

entry = 0
exit = 0

args = get_input_args.parse_user_input()
file_utils.create_paths()

if args['model'] == 'ground-truth':
    input_file = './input_videos/TownCentreXVID.mp4'
else: 
    args['input']

# initialize the video stream, pointer to output video file, and frame dimensions
vs = cv2.VideoCapture(input_file)
fps = int(vs.get(cv2.CAP_PROP_FPS))
total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
(W, H) = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('output/output.mp4', fourcc, fps, (W, H), True)

# get line info
line = image_utils.define_ROI(int(args['line']), input_file, H, W)


if args['model'] == 'ground-truth':
    detector = model_utils.GroundTruthDetections()

    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        detections = detector.get_detected_items(frame_index)
        features, boxes = encoder(frame, detections)

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        tracker.update(detections)

        current = {}
        for track in tracker.tracks:
            d = track.to_tlbr()
            d = np.append(d, track.track_id)
            d = d.astype(np.int32)
            frame = image_utils.draw_box(frame, d, (0,255,0))

            if detections != []:
                cv2.putText(frame, 'Detection active', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            current[d[4]] = (d[0], d[1], d[2], d[3])
            if d[4] in tracker.previous:
                previous_box = tracker.previous[d[4]]
                entry, exit = math_utils.compare_with_prev_position(previous_box, d, line, entry, exit)

        frame = image_utils.annotate_frame(frame, line, entry, exit, H, W)
            
        tracker.previous = current
        writer.write(frame)
        frame_index = frame_index + 1

    writer.release()
    vs.release()

elif args['model'] == 'haar':
    person_cascade = cv2.CascadeClassifier('./detectors/haar_cascade/pedestrian.xml')

    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        detections = model_utils.get_haar_detections(frame, person_cascade, frame_index)
        features, boxes = encoder(frame, detections)

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        tracker.update(detections)

        current = {}
        for track in tracker.tracks:
            d = track.to_tlbr()
            d = np.append(d, track.track_id)
            d = d.astype(np.int32)
            frame = image_utils.draw_box(frame, d, (0,255,0))

            if detections != []:
                cv2.putText(frame, 'Detection active', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            current[d[4]] = (d[0], d[1], d[2], d[3])
            if d[4] in tracker.previous:
                previous_box = tracker.previous[d[4]]
                entry, exit = math_utils.compare_with_prev_position(previous_box, d, line, entry, exit)

        frame = image_utils.annotate_frame(frame, line, entry, exit, H, W)
            
        tracker.previous = current
        writer.write(frame)
        frame_index = frame_index + 1

    writer.release()
    vs.release()

elif args['model'] == 'hog':
    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        detections = model_utils.get_hog_svm_detections(frame, frame_index)
        features, boxes = encoder(frame, detections)

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        tracker.update(detections)

        current = {}
        for track in tracker.tracks:
            d = track.to_tlbr()
            d = np.append(d, track.track_id)
            d = d.astype(np.int32)
            frame = image_utils.draw_box(frame, d, (0,255,0))

            if detections != []:
                cv2.putText(frame, 'Detection active', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            current[d[4]] = (d[0], d[1], d[2], d[3])
            if d[4] in tracker.previous:
                previous_box = tracker.previous[d[4]]
                entry, exit = math_utils.compare_with_prev_position(previous_box, d, line, entry, exit)

        frame = image_utils.annotate_frame(frame, line, entry, exit, H, W)

        tracker.previous = current
        writer.write(frame)
        frame_index = frame_index + 1

    writer.release()
    vs.release()

elif args['model'] == 'yolo':
    labels, net, ln = model_utils.set_yolo_model()

    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # run YOLO object detector
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(ln)

        detections = model_utils.get_yolo_detections(outputs, labels, args['confidence'], args['threshold'], W, H, frame_index)
        features, boxes = encoder(frame, detections)

        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        tracker.update(detections)
        
        current = {}
        for track in tracker.tracks:
            d = track.to_tlbr()
            d = np.append(d, track.track_id)
            d = d.astype(np.int32)
            frame = image_utils.draw_box(frame, d, (0,255,0))

            if detections != []:
                cv2.putText(frame, 'Detection active', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            current[d[4]] = (d[0], d[1], d[2], d[3])
            if d[4] in tracker.previous:
                previous_box = tracker.previous[d[4]]
                entry, exit = math_utils.compare_with_prev_position(previous_box, d, line, entry, exit)

        frame = image_utils.annotate_frame(frame, line, entry, exit, H, W)
    
        tracker.previous = current
        writer.write(frame)
        frame_index = frame_index + 1
    
    writer.release()
    vs.release()


else:
    # load TF detection graph and COCO class labels
    detection_graph, category_index = model_utils.set_tf_model(args['model'], 'mscoco_label_map.pbtxt')

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # for all the frames that are extracted from input video
        frame_index = 0

        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: np.expand_dims(frame, axis=0)}
            )

            detections = model_utils.get_detections(frame, args['confidence'], args['threshold'], boxes, classes, scores, category_index, W, H, frame_index)
            features, boxes = encoder(frame, detections)

            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
            tracker.update(detections)

            current = {}
            for track in tracker.tracks:
                d = track.to_tlbr()
                d = np.append(d, track.track_id)
                d = d.astype(np.int32)
                frame = image_utils.draw_box(frame, d, (0,255,0))

                if detections != []:
                    cv2.putText(frame, 'Detection active', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                current[d[4]] = (d[0], d[1], d[2], d[3])
                if d[4] in tracker.previous:
                    previous_box = tracker.previous[d[4]]
                    entry, exit = math_utils.compare_with_prev_position(previous_box, d, line, entry, exit)

            frame = image_utils.annotate_frame(frame, line, entry, exit, H, W)
            print(frame_index)
            tracker.previous = current
            writer.write(frame)
            frame_index = frame_index + 1

        
        writer.release()
        vs.release()
