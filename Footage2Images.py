import cv2

# This module is to apply the template matching in a video

def fetch_images(footage, start, steps, end):
	frames = []
	for i in range(start):
		footage.grab()
	for i in range(end - start):
		if i % steps == 0:
			frames.append(footage.read())
		else:
			footage.grab()
	return frames
			
video = cv2.VideoCapture("")

video.read()