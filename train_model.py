'''
Image Sensor Vacuum Cleaner 2.
Author:
C. Meyer
'''

import numpy as np
import tensorflow as tf

input = tf.keras.Input((128, 128, 3), name="input")
C1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', kernel_initializer="he_normal", padding="same")(input)
C1 = tf.keras.layers.BatchNormalization()(C1)
C1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', kernel_initializer="he_normal", padding="same")(C1)
C1 = tf.keras.layers.BatchNormalization()(C1)
P1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C1)

C2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', kernel_initializer="he_normal", padding="same")(P1)
C2 = tf.keras.layers.BatchNormalization()(C2)
C2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', kernel_initializer="he_normal", padding="same")(C2)
C2 = tf.keras.layers.BatchNormalization()(C2)
P2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C2)

C3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', kernel_initializer="he_normal", padding="same")(P2)
C3 = tf.keras.layers.BatchNormalization()(C3)
C3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', kernel_initializer="he_normal", padding="same")(C3)
C3 = tf.keras.layers.BatchNormalization()(C3)

C4 = tf.keras.layers.Conv2DTranspose(32, (2, 2), (2, 2), padding='valid')(C3)
C4 = tf.keras.layers.concatenate([C4, C2], axis=3)
C4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal", padding="same")(C4)
C4 = tf.keras.layers.BatchNormalization()(C4)
C4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal", padding="same")(C4)
C4 = tf.keras.layers.BatchNormalization()(C4)

C5 = tf.keras.layers.Conv2DTranspose(32, (2, 2), (2, 2), padding='valid')(C4)
C5 = tf.keras.layers.concatenate([C5, C1], axis=3)
C5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal", padding="same")(C5)
C5 = tf.keras.layers.BatchNormalization()(C5)
C5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal", padding="same")(C5)
C5 = tf.keras.layers.BatchNormalization()(C5)

output = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal", padding="same")(C5)
model = tf.keras.Model(inputs=input, outputs=output)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam())

