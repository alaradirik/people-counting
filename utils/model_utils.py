import os
import tarfile
import urllib

import numpy as np
import cv2
import tensorflow as tf
from imutils.object_detection import non_max_suppression

from utils import label_map_util


class GroundTruthDetections:

    def __init__(self, fname= './ground_truth/TownCentre-groundtruth.top'):
        self.all_dets = np.loadtxt(fname ,delimiter=',') 
        self._frames = int(self.all_dets[:, 1].max()) + 1 

    def perform_detection(self, detect_prob=0.4):
        return int(np.random.choice(2, 1, p=[1 - detect_prob, detect_prob]))
    
    def get_full_data(self, frame):
        return self.all_dets[self.all_dets[:, 1] == frame, :]

    def get_detected_items(self, frame):
        dets = self.all_dets[self.all_dets[:, 1] == frame, 8:]
        dets = np.append(dets, np.ones((dets.shape[0], 1)), axis=-1)
        return dets


    def get_total_frames(self):
        return self._frames


def download_yolo_weights():
    """
    Download pre-trained YOLOv3 model weights.
    """
    download_url = 'https://pjreddie.com/media/files/yolov3.weights'
    opener = urllib.request.URLopener()
    opener.retrieve(download_url, './detectors/yolov3/yolov3.weights')
    return

### TODO: convert YOLOv3 to TF graph
def set_yolo_model():
    """
    Load YOLOv3 object detection model into memory.
    """
    model_found = 0
    MODEL_DIR = './detectors/yolov3'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    files = os.listdir(MODEL_DIR)
    if 'yolov3.weights' in files:
        model_found = 1
    else:
        download_yolo_weights()

    # load model from path
    labels_path = os.path.sep.join([MODEL_DIR, "coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")
    weights_path = os.path.sep.join([MODEL_DIR, "yolov3.weights"])
    config_path = os.path.sep.join([MODEL_DIR, "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return LABELS, net, ln


def set_tf_model(model_name, label_name):

    model_found = 0

    for file in os.listdir("./detectors"):
        if (file == model_name):
            model_found = 1

    model_name = model_name
    model_file = model_name + ".tar.gz"
    download_base = "http://download.tensorflow.org/models/object_detection/"

    # define path to frozen detection graph
    path_to_ckpt = "./detectors/" + model_name + "/frozen_inference_graph.pb"

    # define the list of labels
    path_to_labels = os.path.join("./detectors", label_name)
    num_classes = 90

    # download model if not found
    if (model_found == 0):
        os.makedirs("./detectors/" + model_name)      
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, "./detectors/" + model_file)
        tar_file = tarfile.open("./detectors/" + model_file)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          tar_file.extract(file, "./detectors")
        os.remove("./detectors/" + model_name + ".tar.gz")

    # load frozen Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(path_to_ckpt, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")

    # load label map
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return detection_graph, category_index


def perform_detection(detect_prob=0.4):
    return int(np.random.choice(2, 1, p=[1 - detect_prob, detect_prob]))

def get_haar_detections(frame, cascade, frame_index):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, 1.3, 2)
    
    dets = []
    for (x, y, w, h) in boxes:
        dets.append([x, y, x+w, y+h, 1])
    dets = np.array(dets)
    return dets


def get_hog_svm_detections(frame, frame_index):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (rects, weights) = hog.detectMultiScale(
        frame, 
        winStride=(4, 4),
        padding=(8, 8), 
        scale=1.05
    )

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    dets = []
    for (x, y, x2, y2) in pick:
        dets.append([x, y, x2, y2, 1])
    dets = np.array(dets)
    return dets


def get_yolo_detections(outputs, labels, conf, threshold, W, H, frame_index):
    boxes, confidences, classIDs = [], [], []
    for output in outputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions
            if (confidence > conf) and (labels[classID] == "person"):
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, threshold)

    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, confidences[i]])

    dets = np.array(dets)
    return dets



def get_detections(frame, conf, threshold, boxes, classes, scores, category_index, W, H, frame_index):

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
    idxs = cv2.dnn.NMSBoxes(boxes, scores_person, conf, threshold)

    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x+w, y+h, scores_person[i]])
    dets = np.array(dets)
    return dets



