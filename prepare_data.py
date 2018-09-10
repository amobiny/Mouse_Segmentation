import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(data_dir, name):
    h5f = h5py.File(data_dir + name, 'r')
    x = h5f['x_train'][:]
    y = h5f['y_train'][:]
    h5f.close()
    return x, y


def norm(x, y, mode):
    if mode == 'gaussian':
        pass
    elif mode == 'standard':
        x_max = np.array([np.max(x[:, :, i]) for i in range(x.shape[-1])])
        x_min = np.array([np.min(x[:, :, i]) for i in range(x.shape[-1])])
        x_norm = (x-x_min)/(x_max-x_min)
        y_norm = y/255.
        return x_norm, y_norm


def preprocess(normalize=True, raw_data_dir='./raw_data', out_data_dir='./data'):
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    assert not os.path.exists(raw_data_dir + 'he_sec_a.h5'), 'Please copy the raw data in the raw_data folder'
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    x_train_1, y_train_1 = load_data(raw_data_dir, '/he_sec_a.h5')
    x_train_2, y_train_2 = load_data(raw_data_dir, '/he_sec_b.h5')
    x_test, y_test = load_data(raw_data_dir, '/he_sec_c.h5')
    x_train = np.concatenate((x_train_1[np.newaxis], x_train_2[np.newaxis]), axis=0)
    y_train = np.concatenate((y_train_1[np.newaxis], y_train_2[np.newaxis]), axis=0)

    if normalize:
        x_train, y_train = norm(x_train, y_train, mode='standard')
        x_valid, y_valid = norm(x_test, y_test, mode='standard')
        x_test, y_test = norm(x_test, y_test, mode='standard')
        out_name = out_data_dir+'/data_norm.h5'
    else:
        x_valid = x_test
        y_train = y_train/255.
        y_valid = y_test/255.
        y_test = y_test/255.
        out_name = out_data_dir+'/data.h5'

    h5f = h5py.File(out_name, 'w')
    h5f.create_dataset('x_train', data=x_train.reshape(-1, 512, 512, 60))
    h5f.create_dataset('y_train', data=y_train.reshape(-1, 512, 512, 3))
    h5f.create_dataset('x_valid', data=x_valid)
    h5f.create_dataset('y_valid', data=y_valid)
    h5f.create_dataset('x_test', data=x_test)
    h5f.create_dataset('y_test', data=y_test)
    h5f.close()


if __name__ == '__main__':
    preprocess(normalize=True,
               raw_data_dir='./raw_data',
               out_data_dir='./data')

