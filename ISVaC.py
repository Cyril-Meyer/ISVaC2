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
# np.save("dataset", image_list[:, 0:450, 1350:2250, :])

gradients = np.zeros(image_list.shape)

# Detect anomaly and create mask
print("Compute gradient (Laplacian)")
for frame in trange(image_list.shape[0]):
    gradients[frame] = np.abs(cv2.Laplacian(image_list[frame], cv2.CV_64F))

# TODO : Got better result before meaning in the end, check required

gradients = gradients / gradients.max()
gradient = np.mean(gradients, axis=0)
gradient_1D = np.sum(gradient, axis=-1)/3
mask = (gradient_1D > 0.005)*1.0

open_selem = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_selem)

plt.imshow((gradient > np.quantile(gradient, 0.99)).astype(np.float32))
plt.show()

plt.imshow(mask)
plt.show()

'''
plt.imshow(gradient)
plt.show()

plt.imshow(gradient_1D, cmap='gray')
plt.show()

print((gradient > np.quantile(gradient, 0.99)).astype(np.float32).shape)
plt.imshow((gradient > np.quantile(gradient, 0.99)).astype(np.float32))
plt.show()
'''
