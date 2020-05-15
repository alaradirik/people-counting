import argparse


def parse_user_input():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", help="path to input video", default="./input_videos/TownCentreXVID.mp4")
	ap.add_argument("-o", "--output", help="path to output video", default="./output")
	ap.add_argument("-l", "--line", help="use default ROI or define ROI boundaries", default=0)
	ap.add_argument("-m", "--model", help="base path to detection model", default="yolo")
	ap.add_argument("-d", "--dlib", help="base path to detection model", default=True)
	ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")

	args = vars(ap.parse_args())
	return args