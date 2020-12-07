from random import randint
import numpy as np
import tensorflow as tf
import segmentation_models as sm
from patches import patches


train_images = np.load("train_images.npy")
valid_images = np.load("valid_images.npy")

train = patches(128, train_images, batch_size=8)
valid = patches(128, valid_images, batch_size=8)

model = sm.Unet('seresnext101', classes=3, activation='linear', encoder_weights=None)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam())
'''
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.Adam())
'''

model.fit(train, epochs=50, steps_per_epoch=512, validation_data=valid, validation_steps=64)
'''
model.fit(train, epochs=50, steps_per_epoch=512, validation_data=valid, validation_steps=64,
          callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="model_MeanSquaredError", save_best_only=True)])
model.fit(train, epochs=50, steps_per_epoch=512, validation_data=valid, validation_steps=64,
          callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="model_Huber", save_best_only=True)])
'''

tf.keras.models.save_model(model, "model_MeanSquaredError")
# tf.keras.models.save_model(model, "model_Huber")
