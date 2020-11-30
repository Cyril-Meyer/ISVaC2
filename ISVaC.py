'''
Image Sensor Vacuum Cleaner 2.
Author:
C. Meyer
'''

import os

import numpy as np
import cv2

from tqdm import tqdm, trange

import matplotlib.pyplot as plt


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


image_list = []
files_list = []
file_names_list = []
files_number = 0

# Read file names
for file in os.listdir("./data"):
    if file.endswith(".JPG"):
        files_list.append(os.path.join("./data", file))
        file_names_list.append(file)
        files_number += 1
    '''
    if file.endswith(".AVI"):
        files_list.append(os.path.join("./data", file))
        file_names_list.append(file)
        files_number += 1
    '''

# Read images/videos
print("Read Files")
for filename in tqdm(files_list):
    if filename.endswith(".JPG"):
        image = np.array(cv2.imread(filename, cv2.IMREAD_COLOR), dtype=np.uint8)
        image_list.append(np.expand_dims(image, 0))
    elif filename.endswith(".AVI"):
        video = avi_to_np(filename)
        image_list.append(video)

# Convert image list in numpy array for numba
image_list = np.concatenate(image_list)

# Get dataset
np.save("dataset.npy", image_list[:, 0:1200, 200:600, :]/255.0)

gradients = np.zeros(image_list.shape)

# Detect anomaly and create mask
'''
print("Compute RGB gradient (Laplacian)")
for frame in trange(image_list.shape[0]):
    gradients[frame] = np.abs(cv2.Laplacian(image_list[frame], cv2.CV_64F))
'''
'''
print("Compute RGB gradient (Scharr)")
for frame in trange(image_list.shape[0]):
    gradients[frame] = (cv2.Scharr(image_list[frame], cv2.CV_64F, 1, 0))
    gradients[frame] += (cv2.Scharr(image_list[frame], cv2.CV_64F, 0, 1))
'''
print("Compute gradient (Laplacian)")
gradients = np.zeros((24, 1920, 2560))
for frame in trange(image_list.shape[0]):
    gradients[frame] = np.abs(cv2.Laplacian(cv2.cvtColor(image_list[frame], cv2.COLOR_BGR2GRAY), cv2.CV_64F))

gradients = (gradients - gradients.min()) / gradients.max()

if gradients.ndim == 4:
    gradients = np.sum(gradients, axis=-1)/3

'''
gradient = np.std(gradients, axis=0)
mask = (gradient > 0.015)*1.0
'''
gradient = np.mean(gradients, axis=0)
mask = (gradient > 0.010)*1.0


selem_3 = np.array([[0, 1,  0],
                    [1, 1,  1],
                    [0, 1,  0]], dtype=np.uint8)

selem_5 = np.array([[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]], dtype=np.uint8)

mask = cv2.erode(mask, selem_3)
mask = cv2.dilate(mask, selem_3)
mask = cv2.dilate(mask, selem_5)
mask = cv2.dilate(mask, selem_5)
mask = cv2.dilate(mask, selem_5)
mask = cv2.erode(mask, selem_5)


plt.imshow(mask, cmap='gray')
plt.show()
