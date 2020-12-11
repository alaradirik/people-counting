# Object Counter

University of Glasgow, MSc. Thesis Project

This project consists of a set of experiments to compare the performance of combinations of popular and state-of-the-art object detection models and multi-object tracking algorithms. All combinations are evaluated on the [Town Centre Dataset](http://www.robots.ox.ac.uk/~lav/Research/Projects/2009bbenfold_headpose/project.html) and the [MOT17](https://motchallenge.net/data/MOT17/) dataset with the objective of tracking and counting the pedestrians crossing a LOI in a video sequence (MOT). New evaluation datasets will be added.


## Project Structure

    .
    ├── detectors/                      # Pre-trained person detection models files and frozen TF graphs
    ├── trackers/                       # Multi-object tracking algorithms
    ├── utils                           # Utility functions to process inputs and video frames
        └── file_utils.py
        └── image_utils.py
        └── model_utils.py                          
        └── math_utils.py
        └── label_map_util.py                  
        └── get_input_args.py
    ├── protos/                         # Tensorflow Object Detection API protos
    ├── input_videos/                   # Input videos
    ├── output/                         # Output videos                         
    ├── ground_truth                    # Ground truth data for the Oxford Town Centre Dataset
    ├── run_sort.py                     # Runs SORT
    ├── run_deepsort.py                 # Runs DeepSORT
    ├── groundtruth.py                  # Generates Oxford Town Centre video with ground truth annotations
    ├── requirements.txt                # Dependencies
    └── README.md

## Installation

Input arguments and options:
- **input:** path to input video
- **output:** path to output folder
- **model:** object detection model 
- **dlib:** use kernelized correlation filter or Kalman Filter - True or False 
- **confidence:** confidence threshold for object detection - default is 0.5 
- **threshold:** non-maxima suppression threshold - default is 0.3
- **line:** automatically assign ROI, use user input to draw a boundary line on frame or manually enter line coordinates - select 0, 1 or 2

--line 1 opens a visual interface to draw a line on the input video, --line 2 prompts user to enter line coordinates: x1 y1 x2 y2.

Run the script:
python run_sort.py --model [DETECTION_MODEL]--input [INPUT_VIDEO_PATH] --output [OUTPUT_FOLDER] --line 0  
python run_deepsort.py --model [DETECTION_MODEL]--input [INPUT_VIDEO_PATH] --output [OUTPUT_FOLDER] --line 0

## Available Detection Models
- YOLOv3 can be loaded using the model argument: --model yolo
- TensorFlow models pretrained on the COCO dataset. A full list of the available models can be seen at [Tensorflow Model Zoo.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- HAAR Cascade for full body detection: --model haar
- HOG with Linear SVM: --model hog

To use the TensorFlow models select a model from the model zoo and pass it as the model argument: --model ssdlite_mobilenet_v2_coco. This automatically creates a folder in detectors/ and downloads the model files if theey don't exist.

Use --model ground-truth to track with ground truth detections. ground-truth option uses the annotations of the [Town Centre Dataset](https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html).

## Available Trackers
- [SORT](https://arxiv.org/abs/1602.00763) with Kalman Filter
- SORT with Correlation Filter
- [DeepSORT](https://arxiv.org/abs/1703.07402)

DeepSORT code is largely borrowed from Nikolai Wojke's [DeepSORT repo](https://github.com/nwojke/deep_sort).
