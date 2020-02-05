import os
import tarfile
import urllib

import cv2
import tensorflow as tf

from utils import label_map_util


def download_yolo_weights():
    """
    Download pre-trained YOLOv3 model weights.
    """
    download_url = "https://pjreddie.com/media/files/yolov3.weights"
    print("downloading model weights...")

    opener = urllib.request.URLopener()
    opener.retrieve(download_url, "./models/yolo-obj/yolov3.weights")
    return


def set_yolo_model():
    """
    Load YOLOv3 object detection model into memory.
    """
    model_found = 0
    files = os.listdir("./models/yolo-obj")

    if "yolov3.weights" in files:
        model_found = 1
    else:
        download_yolo_weights()

    # load model from path
    labels_path = os.path.sep.join(["./models/yolo-obj", "coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")
    weights_path = os.path.sep.join(["./models/yolo-obj", "yolov3.weights"])
    config_path = os.path.sep.join(["./models/yolo-obj", "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # determine only the output layer names
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return LABELS, net, ln


def set_model(model_name, label_name):

    model_found = 0

    for file in os.listdir("./models"):
        if (file == model_name):
            model_found = 1

    model_name = model_name
    model_file = model_name + ".tar.gz"
    download_base = "http://download.tensorflow.org/models/object_detection/"

    # define path to frozen detection graph
    path_to_ckpt = "./models/" + model_name + "/frozen_inference_graph.pb"

    # define the list of labels
    path_to_labels = os.path.join("./models", label_name)
    num_classes = 90

    # download model if not found
    if (model_found == 0):      
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, "./models/" + model_file)
        tar_file = tarfile.open("./models/" + model_file)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          tar_file.extract(file, "./models")

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
