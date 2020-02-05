# Object Counter

University of Glasgow, MSc. Thesis Project

## Project Structure

    .
    ├── models                          # Pre-trained model files and frozen TF graphs
        └── yolo-obj                    # YOLOv3
        └── faster-rcnn                 # Faster-RCNN Inception
        └── ssd-mobilenet               # SSD Mobilenet
        └── ssd-inception               # SSD Inception
    ├── utils                           # Utility functions to process inputs and video frames
        └── image_utils.py
        └── draw_line.py                # Interface to draw user defined ROI                 
        └── math_utils.py
        └── label_map_util.py                  
        └── process_input.py
    ├── protos/                         # Tensorflow Object Detection API protos
    ├── input/                          # Input videos
    ├── output/                         # Output videos                         
    ├── sort.py                         # SORT object tracker
    ├── main.py                         # Functions to run object detector and counter
    ├── requirements.txt                # Dependencies
    └── README.md

## Installation

Input arguments and options:
- **input:** path to input video
- **output:** path to output folder
- **model:** object detector model - available models can be found at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
- **confidence:** confidence threshold for object detection - default is 0.5
- **threshold:** non-maxima suppression threshold - default is 0.3
- **line:** automatically assign ROI or use user input to draw 2 boundary lines - select 0 or 1

Run the script:
python main.py --input [INPUT_VIDEO_PATH] --output [OUTPUT_FOLDER] --line 0

## References
### YOLOv3 :

    @article{yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Redmon, Joseph and Farhadi, Ali},
    journal = {arXiv},
    year={2018}
    }

### SORT :

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}