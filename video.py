import numpy as np
import cv2


def avi_to_np(avi_filename):
    video_cv2 = cv2.VideoCapture(avi_filename)
    video = []

    read_state = 0
    while True:
        ret, data = video_cv2.read()
        if not ret:
            break
        video.append(data)

    video_cv2.release()

    return np.array(video, dtype=np.uint8)
