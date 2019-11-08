# Object Counter

University of Glasgow, MSc. Thesis Project

## Project Structure

    .
    ├── yolo-obj                        # Pre-trained model files
        └── coco.names                  # Object labels (person, car, etc.)
        └── yolov3.cfg                  # Model configuration
        └── yolov3.weights              # Model weights
    ├── utils                           # Utility functions to process inputs and video frames
        └── image_utils.py                 
        └── math_utils.py                  
        └── process_input.py                         
    ├── sort.py                         # SORT object tracker
    ├── main.py                         # Functions to run object detector and counter
    ├── requirements.txt                # Dependencies
    └── README.md

## Installation

Input arguments and options:
--input:        path to input video
--output:       path to output folder
--model:        object detector model - default and only available model is YOLO for now
--confidence:   confidence threshold for object detection - default is 0.5
--threshold:    non-maxima suppression threshold - default is 0.3
--line:         automatically assign ROI or use user input to draw 1 or 2 boundary lines - select 0,1 or 2

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