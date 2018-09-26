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
            self.file_name = 'DAPI_kid_crop.h5'
        else:
            self.file_name = 'data_abc.h5'
        self.height, self.width = cfg.height, cfg.width
        #self.max_bottom_left_front_corner = (cfg.img_size - cfg.height - 1, cfg.img_size - cfg.width - 1)
        # maximum value that the bottom left front corner of a cropped patch can take

    def next_batch(self, start=None, end=None, mode='train'):
        h5f = h5py.File(self.data_dir + self.file_name, 'r')
        if mode == 'train':
            img_size = h5f['x_train'].shape
            self.max_bottom_left_front_corner = (img_size[1] - self.cfg.height - 1, img_size[2] - self.cfg.width - 1)
            img_idxs = np.random.choice(self.cfg.num_tr, replace=True, size=self.cfg.batch_size)

            if self.cfg.mask:
                idxs = self.idxs
                np.random.shuffle(idxs)
                bottom_coords = idxs[0:self.cfg.batch_size, 0]
                left_coords = idxs[0:self.cfg.batch_size, 1]
                #bottom_coords = self.idxs[:,0]
                #left_coords = self.idxs[:,1]
            else:
                bottom_coords = np.random.randint(self.max_bottom_left_front_corner[0], size=self.cfg.batch_size) #
                left_coords = np.random.randint(self.max_bottom_left_front_corner[1], size=self.cfg.batch_size)

            x = np.array([h5f['x_train'][img_idx, bottom:bottom + self.height, left:left + self.width, :]
                          for img_idx, bottom, left in zip(img_idxs, bottom_coords, left_coords)])
            y = np.array([h5f['y_train'][img_idx, bottom:bottom + self.height, left:left + self.width, :]
                          for img_idx, bottom, left in zip(img_idxs, bottom_coords, left_coords)])
            if self.cfg.data_augment:
                x, y = random_flip_2d(x, y)
            return x, y
        elif mode == 'valid':
            return self.x_valid[start:end], self.y_valid[start:end]
        elif mode == 'test':
            return self.x_test[start:end], self.y_test[start:end]

    def get_data(self, mode='valid'):
        if mode == 'valid':
            h5f = h5py.File(self.data_dir + self.file_name, 'r')
            self.val_img_size = h5f['x_valid'].shape
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
    return img_batch_rot, mask_batch_rot


def rotate(x, angle):
    return scipy.ndimage.interpolation.rotate(x, angle, mode='nearest', axes=(0, 1), reshape=False)


def random_flip_2d(img_batch, mask_batch):
    """
    Randomly flip an image left to right or up and down
    :param img_batch: batch of 3D images
    :param mask_batch: batch of 3D masks
    :return: batch of flipped 3D images and masks
    """
    img_batch_flip, mask_batch_flip = img_batch, mask_batch
    for i in range(img_batch.shape[0]):
        image, mask = img_batch[i], mask_batch[i]
        if np.random.randint(2):
            img_batch_flip[i] = np.fliplr(image)
            mask_batch_flip[i] = np.fliplr(mask)
        if np.random.randint(2):
            img_batch_flip[i] = np.flipud(image)
            mask_batch_flip[i] = np.flipud(mask)
    return img_batch_flip, mask_batch_flip






