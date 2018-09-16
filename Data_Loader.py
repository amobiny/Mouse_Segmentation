import random
import numpy as np
import h5py
import scipy.ndimage
from math import floor
import matplotlib.pyplot as plt

class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        if cfg.normalize:
            self.file_name = 'data_norm.h5'
        else:
            self.file_name = 'data.h5'
        self.height, self.width = cfg.height, cfg.width
        self.max_bottom_left_front_corner = (cfg.img_size - cfg.height - 1, cfg.img_size - cfg.width - 1)
        # maximum value that the bottom left front corner of a cropped patch can take

    def next_batch(self, start=None, end=None, mode='train'):
        h5f = h5py.File(self.data_dir + self.file_name, 'r')
        if mode == 'train':
            img_idxs = np.random.choice(self.cfg.num_tr, replace=True, size=self.cfg.batch_size)
            bottom_coords = np.random.randint(self.max_bottom_left_front_corner[1], size=self.cfg.batch_size) #
            left_coords = np.random.randint(self.max_bottom_left_front_corner[0], size=self.cfg.batch_size)
            x = np.array([h5f['x_train'][img_idx, bottom:bottom + self.height, left:left + self.width, :]
                          for img_idx, bottom, left in zip(img_idxs, bottom_coords, left_coords)])
            y = np.array([h5f['y_train'][img_idx, bottom:bottom + self.height, left:left + self.width, :]
                          for img_idx, bottom, left in zip(img_idxs, bottom_coords, left_coords)])
            if self.cfg.data_augment:
                x, y = random_rotation_2d(x, y, max_angle=self.cfg.max_angle)
            return x, y
        elif mode == 'valid':
            return self.x_valid[start:end], self.y_valid[start:end]
        elif mode == 'test':
            return self.x_test[start:end], self.y_test[start:end]

    def get_data(self, mode='valid'):
        if mode == 'valid':
            h5f = h5py.File(self.data_dir + self.file_name, 'r')
            x_valid = h5f['x_valid'][:]
            y_valid = h5f['y_valid'][:]
            self.x_valid = np.expand_dims(x_valid, 0)
            self.y_valid = np.expand_dims(y_valid, 0)
        if mode == 'test':
            h5f = h5py.File(self.data_dir + self.file_name, 'r')
            x_test = h5f['x_test'][:]
            y_test = h5f['y_test'][:]
            self.x_test = np.expand_dims(x_test, 0)
            self.y_test = np.expand_dims(y_test, 0)

def random_rotation_2d(img_batch, mask_batch, max_angle):
    """
    Randomly rotate an image by a random angle (-max_angle, max_angle)
    :param img_batch: batch of 3D images
    :param mask_batch: batch of 3D masks
    :param max_angle: `float`. The maximum rotation angle
    :return: batch of rotated 3D images and masks
    """
    img_batch_rot, mask_batch_rot = img_batch, mask_batch
    for i in range(img_batch.shape[0]):
        image, mask = img_batch[i], mask_batch[i]
        angle = random.uniform(-max_angle, max_angle)
        img_batch_rot[i] = rotate(image, angle)
        mask_batch_rot[i] = rotate(mask, angle)
    return img_batch_rot, mask_batch


def rotate(x, angle):
    return scipy.ndimage.interpolation.rotate(x, angle, mode='nearest', axes=(0, 1), reshape=False)



