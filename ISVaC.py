'''
Image Sensor Vacuum Cleaner 2.
Author:
C. Meyer
'''

import os

import numpy as np
import cv2
import tensorflow as tf

from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from video import avi_to_np


image_list = []
files_list = []
file_names_list = []
files_number = 0

# Read input files --------------------------------------------------------------------------------------------------- #
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
        image_list.append(np.expand_dims(np.array(cv2.imread(filename, cv2.IMREAD_COLOR), dtype=np.uint8), 0))
    elif filename.endswith(".AVI"):
        image_list.append(avi_to_np(filename))

# Convert image list in numpy array
image_list = np.concatenate(image_list)

# Save training and validation data.
# np.save("train_images.npy", image_list[:, 0:1200, 200:600, :]/255.0)
# np.save("valid_images.npy", image_list[:, 0:300, 1900:2200, :]/255.0)

# Create mask -------------------------------------------------------------------------------------------------------- #
# Compute gradients for each image
gradients = np.zeros(image_list.shape)
'''
print("Compute RGB gradient (Laplacian)")
for frame in trange(image_list.shape[0]):
    gradients[frame] = np.abs(cv2.Laplacian(image_list[frame], cv2.CV_64F))
'''
'''
print("Compute RGB gradient (Scharr)")
for frame in trange(image_list.shape[0]):
    gradients[frame] = np.abs(cv2.Scharr(image_list[frame], cv2.CV_64F, 1, 0))
    gradients[frame] += np.abs(cv2.Scharr(image_list[frame], cv2.CV_64F, 0, 1))
'''
print("Compute gradient (Laplacian)")
gradients = np.zeros((image_list.shape[0], 1920, 2560))
for frame in trange(image_list.shape[0]):
    gradients[frame] = np.abs(cv2.Laplacian(cv2.cvtColor(image_list[frame], cv2.COLOR_BGR2GRAY), cv2.CV_64F))

# Scale gradients
gradients = (gradients - gradients.min()) / gradients.max()
# Merge RGB gradients if needed
if gradients.ndim == 4:
    gradients = np.sum(gradients, axis=-1)/3
# Mean gradients, convert them into a mask
gradient = np.mean(gradients, axis=0)
mask = (gradient > 0.010)*1.0

# Better mask -------------------------------------------------------------------------------------------------------- #
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
mask = cv2.erode(mask, selem_5)

# Reconstruction ----------------------------------------------------------------------------------------------------- #
model = tf.keras.models.load_model("model")

model.summary()

image_list = image_list / 255.0
plt.imshow(mask)
for frame in trange(image_list.shape[0]):
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] == 1.0:
                image_list[frame, x, y, :] = -1

    plt.imshow(model.predict(np.expand_dims(image_list[frame, 0:128, 0:128], axis=0)))
    plt.show()
# cv2.imwrite(filename, img)
