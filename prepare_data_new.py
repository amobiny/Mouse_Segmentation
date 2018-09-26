import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


def load_data(data_dir, name,x_name,y_name):
    h5f = h5py.File(data_dir + name, 'r')
    x = h5f[x_name][:]
    y = h5f[y_name][:]
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
    assert not os.path.exists(raw_data_dir + 'he_ab.h5'), 'Please copy the raw data in the raw_data folder'
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    x_name = 'x_train'
    y_name = 'y_train'
    x_train, y_train = load_data(raw_data_dir, '/he_ab.h5',x_name,y_name)

    x_name = 'x_test'
    y_name = 'y_test'
    x_test, y_test = load_data(raw_data_dir, '/he_c.h5',x_name,y_name)


    if normalize:
        x_train, y_train = norm(x_train, y_train, mode='standard')
        x_valid, y_valid = norm(x_test, y_test, mode='standard')
        x_test, y_test = norm(x_test, y_test, mode='standard')
        out_name = out_data_dir+'/data_norm1.h5'
    else:
        x_valid = x_test
        y_train = y_train/255.
        y_valid = y_test/255.
        y_test = y_test/255.
        out_name = out_data_dir+'/data1.h5'

    x_train = x_train[np.newaxis]
    y_train = y_train[np.newaxis]

    h5f = h5py.File(out_name, 'w')
    h5f.create_dataset('x_train', data=x_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('x_valid', data=x_valid)
    h5f.create_dataset('y_valid', data=y_valid)
    h5f.create_dataset('x_test', data=x_test)
    h5f.create_dataset('y_test', data=y_test)
    h5f.close()


if __name__ == '__main__':
    preprocess(normalize=True,
               raw_data_dir='./raw_data',
               out_data_dir='./data')

