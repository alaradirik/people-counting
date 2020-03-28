import os

def create_paths():
	if not os.path.exists('output'):
		os.makedirs('output')

	if not os.path.exists('input_videos'):
		os.makedirs('input_videos')