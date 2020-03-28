# Object Counter

University of Glasgow, MSc. Thesis Project

This project an experiment to compare the performance of combinations of popular and state-of-the-art object detection models and multi-object tracking algorithms. All combinations are evaluated on the [Oxford Town Centre Dataset](http://www.robots.ox.ac.uk/~lav/Research/Projects/2009bbenfold_headpose/project.html) with the objective of tracking and counting multiple pedestrians crossing a ROI in a video sequence (MOT).

## Project Structure

    .
    ├── detector/                       # Pre-trained person detection models files and frozen TF graphs
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
    ├── main.py                         # Runs the pedestrian counting app
    ├── groundtruth.py                  # Generates Oxford Town Centre video with ground truth annotations
    ├── track_with_groundtruth.py       # Track using ground truth detections (for Oxford Town Centre video)
    ├── requirements.txt                # Dependencies
    └── README.md

## Installation

Input arguments and options:
- **input:** path to input video
- **output:** path to output folder
- **model:** object detection model 
- **confidence:** confidence threshold for object detection - default is 0.5 
- **threshold:** non-maxima suppression threshold - default is 0.3
- **line:** automatically assign ROI, use user input to draw a boundary line on frame or manually enter line coordinates - select 0, 1 or 2

Run the script:
python main.py --model [DETECTION_MODEL]--input [INPUT_VIDEO_PATH] --output [OUTPUT_FOLDER] --line 0

## Available Detection Models
- YOLOv3 can be loaded using the model argument: --model yolo
- Tensorflow models pretrained on the COCO dataset. A full list of the available models can be seen at [Tensorflow Model Zoo.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- HAAR Cascade for full body detection: --model haar

## Available Trackers
- SORT with Kalman Filter
- SORT with Correlation Filter

## References
#### YOLOv3 :

    @article{yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Redmon, Joseph and Farhadi, Ali},
    journal = {arXiv},
    year={2018}
    }

#### SORT :

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}

