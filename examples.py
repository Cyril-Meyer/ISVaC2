import os
from random import randint
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from patches import patches

images = np.load("train_images.npy")
train = patches(128, images, batch_size=16, remove=10)
# train = patches(64, images, batch_size=16, remove=4)

X, Y = next(train)

fig = plt.figure(figsize=(8, 8))
fig.add_subplot(2, 2, 1)
plt.imshow(np.clip(X[0], 0, np.inf))
fig.add_subplot(2, 2, 2)
plt.imshow(np.clip(Y[0], 0, np.inf))
fig.add_subplot(2, 2, 3)
plt.imshow(np.clip(X[1], 0, np.inf))
fig.add_subplot(2, 2, 4)
plt.imshow(np.clip(Y[1], 0, np.inf))
plt.show()

model = tf.keras.models.load_model("model")

Z = model.predict(X)

fig = plt.figure(figsize=(8, 8))
fig.add_subplot(2, 2, 1)
plt.imshow(np.clip(X[0], 0, np.inf))
fig.add_subplot(2, 2, 2)
plt.imshow(np.clip(Z[0], 0, np.inf))
fig.add_subplot(2, 2, 3)
plt.imshow(np.clip(X[1], 0, np.inf))
fig.add_subplot(2, 2, 4)
plt.imshow(np.clip(Z[1], 0, np.inf))
plt.show()
