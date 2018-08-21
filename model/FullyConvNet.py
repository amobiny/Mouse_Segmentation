import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_2d


class FCN(BaseModel):
    def __init__(self, sess, conf):
        super(FCN, self).__init__(sess, conf)
        self.act_fcn = tf.nn.relu
        self.k_size = self.conf.filter_size
        self.pool_size = self.conf.pool_filter_size
        self.build_network()
        self.configure_network()

    def build_network(self):
        # Building network...
        with tf.variable_scope('FCN'):
            conv1 = conv_2d(self.x, self.k_size, 64, 'CONV1', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv2 = conv_2d(conv1, self.k_size, 128, 'CONV2', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv3 = conv_2d(conv2, self.k_size, 256, 'CONV3', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv4 = conv_2d(conv3, self.k_size, 128, 'CONV4', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv5 = conv_2d(conv4, self.k_size, 64, 'CONV5', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv6 = conv_2d(conv5, self.k_size, 32, 'CONV6', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv7 = conv_2d(conv6, self.k_size, 16, 'CONV7', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            self.logits = conv_2d(conv7, 1, self.conf.out_channel, 'CONV8', batch_norm=False,
                                  is_train=self.is_training)
