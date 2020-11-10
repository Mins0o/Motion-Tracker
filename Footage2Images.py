import cv2

# This module is to apply the template matching in a video

def fetch_images(footage, start, steps, end):

video = cv2.VideoCapture("")

video.read()