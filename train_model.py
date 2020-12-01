from random import randint
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from patches import patches

train_images = np.load("train_images.npy")
valid_images = np.load("valid_images.npy")

train = patches(128, train_images, batch_size=16)
valid = patches(128, valid_images, batch_size=16)

model = sm.Unet('seresnet50', classes=3, activation='linear', encoder_weights=None)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam())

model.fit(train, epochs=25, steps_per_epoch=256, validation_data=valid, validation_steps=64)

tf.keras.models.save_model(model, "model")
