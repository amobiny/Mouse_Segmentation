import h5py
import numpy as np
import matplotlib.pyplot as plt

raw_data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/Mouse_Segmentation/raw_data'


def load_data(data_dir, name):
    h5f = h5py.File(data_dir + name, 'r')
    x = h5f['x_train'][:]
    y = h5f['y_train'][:]
    h5f.close()
    return x, y


x_train, y_train = load_data(raw_data_dir, '/he_sec_a.h5')
x_valid, y_valid = load_data(raw_data_dir, '/he_sec_b.h5')
x_test, y_test = load_data(raw_data_dir, '/he_sec_c.h5')

print()




