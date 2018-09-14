import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_2d , max_pool , fc_layer, flatten_layer

class CNN(BaseModel):
    def __init__(self , sess, conf):
        super(CNN, self).__init__(sess, conf)
        self.act_fcn = tf.nn.softplus
        self.k_size = self.conf.filter_size
        self.pool_size = self.conf.pool_filter_size
        self.build_network()
        self.configure_network()

    def build_network(self):
        with tf.variable_scope('CNN'):
            conv1 = conv_2d(self.x, self.k_size, 32, 'CONV1', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv2 = conv_2d(conv1, self.k_size, 32, 'CONV2', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            maxpool1 = max_pool(conv2, self.pool_size, 'maxpool1')
            conv3 = conv_2d(maxpool1, self.k_size, 64, 'CONV3', batch_norm=self.conf.use_BN,
                            is_train=self.is_training, activation=self.act_fcn)
            conv4 = conv_2d(conv3, self.k_size, 64, 'CONV4', batch_norm=self.conf.use_BN,
                    is_train=self.is_training, activation=self.act_fcn)
            maxpool2 = max_pool(conv4, self.pool_size, 'maxpool2')
            layer_flat = flatten_layer(maxpool2)

            fc1= fc_layer(layer_flat , 128, name='fc1', activation=self.act_fcn)
            fc2 = fc_layer(fc1 , 3, 'fc2')
            self.logits = fc2