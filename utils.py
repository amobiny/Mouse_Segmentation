import tensorflow as tf
import tensorlayer as tl
import numpy as np
from sklearn.metrics import confusion_matrix


def get_num_channels(x):
    """
    returns the input's number of channels
    :param x: input tensor with shape [batch_size, ..., num_channels]
    :return: number of channels
    """
    return x.get_shape().as_list()[-1]


def add_noise(batch, mean=0, var=0.1, amount=0.01, mode='pepper'):
    original_size = batch.shape
    batch = np.squeeze(batch)
    batch_noisy = np.zeros(batch.shape)
    for ii in range(batch.shape[0]):
        image = np.squeeze(batch[ii])
        if mode == 'gaussian':
            gauss = np.random.normal(mean, var, image.shape)
            image = image + gauss
        elif mode == 'pepper':
            num_pepper = np.ceil(amount * image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        elif mode == "s&p":
            s_vs_p = 0.5
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        batch_noisy[ii] = image
    return batch_noisy.reshape(original_size)


def write_spec(args):
    file = open(args.modeldir + args.run_name +'/config.txt', 'w')
    file.write('model: '+args.run_name+'\n')
    file.write('num_cls: ' + str(args.num_cls)+'\n')
    file.write('optimizer: '+'Adam'+'\n')
    file.write('learning_rate: '+str(args.init_lr) + ' : ' + str(args.lr_min) +'\n')
    file.write('loss_type: '+args.loss_type+'\n')
    file.write('batch_size: '+str(args.batch_size)+'\n')
    file.write('data_augmentation: '+str(args.data_augment)+'\n')
    file.write('    max_angle: '+ str(args.max_angle)+'\n')
    file.write('num_training: '+ str(args.num_tr)+'\n')
    file.write('drop_out_rate: '+str(args.drop_out_rate)+'\n')
    file.write('batch_normalization: '+ str(args.use_BN)+ '\n')
    file.write('kernel_size: '+str(args.filter_size)+ '\n')
    file.close()
