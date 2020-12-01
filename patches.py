from random import randint
import numpy as np


def patches(patch_size, image, batch_size=32, remove=10):
    batch_image = np.zeros((batch_size, patch_size, patch_size, 3))
    batch_label = np.zeros((batch_size, patch_size, patch_size, 3))
    while True:
        for i in range(batch_size):
            x = randint(0, image.shape[2] - patch_size - 1)
            y = randint(0, image.shape[1] - patch_size - 1)
            z = randint(0, image.shape[0] - 1)

            batch_label[i, :, :, :] = image[z, y:y + patch_size, x:x + patch_size]
            batch_image[i, :, :, :] = image[z, y:y + patch_size, x:x + patch_size]

            # Removed part = -1
            for j in range(remove):
                s = randint(2, 16)
                x_ = randint(0, patch_size-s)
                y_ = randint(0, patch_size-s)

                batch_image[i, y_:y_+s, x_:x_+s, :] = -1

            # Augmentations
            # random 90 degree rotation
            # random flip
            rot = randint(0, 3)
            batch_image[i, :, :] = np.rot90(batch_image[i, :, :], rot)
            batch_label[i, :, :] = np.rot90(batch_label[i, :, :], rot)

            if randint(0, 1) == 1:
                batch_image[i, :, :] = np.fliplr(batch_image[i, :, :])
                batch_label[i, :, :] = np.fliplr(batch_label[i, :, :])

            if randint(0, 1) == 1:
                batch_image[i, :, :] = np.flipud(batch_image[i, :, :])
                batch_label[i, :, :] = np.flipud(batch_label[i, :, :])

        yield batch_image, batch_label
