import math
import numpy as np


def distance(true_boxes, det_boxes):
    # Computes the Eucledian distance between ground truth and hypothesis boxes
    dist_matrix = []

    for true_box in true_boxes:
        obj_to_hyps = []
        xg = true_box[2] - true_box[1] / 2
        yg = true_box[3] - true_box[0] / 2

        for det_box in det_boxes:
            xdg = det_box[2] - det_box[1] / 2
            ydg = det_box[3] - det_box[0] / 2
            
            dist = math.sqrt((xg-xdg)**2 + (yg-ydg)**2)
            obj_to_hyps.append(dist)

        dist_matrix.append(obj_to_hyps)
    return dist_matrix
