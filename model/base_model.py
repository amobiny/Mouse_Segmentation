import tensorflow as tf
from Data_Loader import DataLoader
import os
import numpy as np

# add a line for pull test
class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.is_training = True
        self.input_shape = [None, self.conf.height, self.conf.width, self.conf.in_channel]
        self.output_shape = [None, self.conf.height, self.conf.width, self.conf.out_channel]
        self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.float32, self.output_shape, name='annotation')
            self.keep_prob = tf.placeholder(tf.float32)

    def loss_func(self):
        with tf.name_scope('Loss'):
            if self.conf.loss_type == 'MSE':
                with tf.name_scope('MSE'):
                    self.loss = tf.norm(self.y - self.logits)
            if self.conf.use_reg:
                with tf.name_scope('L2_loss'):
                    l2_loss = tf.reduce_sum(
                        self.conf.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('reg_weights')]))
                    self.loss += l2_loss
            self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.loss)

    def configure_network(self):
        self.y_pred = tf.sigmoid(self.logits)
        self.loss_func()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   global_step,
                                                   decay_steps=500,
                                                   decay_rate=0.97,
                                                   staircase=True)
        self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        summary_list = [tf.summary.scalar('learning_rate', self.learning_rate),
                        tf.summary.scalar('loss', self.mean_loss),
                        tf.summary.image('prediction_mask', self.y_pred, max_outputs=3),
                        tf.summary.image('original_mask', self.y, max_outputs=3)]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step):
        if self.is_training:
            self.train_writer.add_summary(summary, step)
        else:
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
        else:
            print('----> Start Training')
        self.best_validation_loss = 10000
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='valid')
        self.num_val_batch = int(self.data_reader.y_valid.shape[0] / self.conf.batch_size)
        for train_step in range(1, self.conf.max_step + 1):
            self.is_training = True
            if train_step % self.conf.SUMMARY_FREQ == 0:
                x_batch, y_batch = self.data_reader.next_batch(mode='train')
                feed_dict = {self.x: x_batch, self.y: y_batch, self.keep_prob: self.conf.drop_out_rate}
                _, _, summary = self.sess.run([self.train_op,
                                               self.mean_loss_op,
                                               self.merged_summary],
                                              feed_dict=feed_dict)
                loss = self.sess.run(self.mean_loss)
                print('step: {0:<6}, train_loss= {1:.4f}'.format(train_step, loss))
                self.save_summary(summary, train_step + self.conf.reload_step)
            else:
                x_batch, y_batch = self.data_reader.next_batch(mode='train')
                feed_dict = {self.x: x_batch, self.y: y_batch, self.keep_prob: self.conf.drop_out_rate}
                self.sess.run([self.train_op, self.mean_loss_op], feed_dict=feed_dict)
            if train_step % self.conf.VAL_FREQ == 0:
                self.is_training = False
                self.evaluate(train_step)

    def evaluate(self, train_step):
        self.is_training = False
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_val_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.x: x_val, self.y: y_val, self.keep_prob: 1}
            self.sess.run(self.mean_loss_op, feed_dict=feed_dict)
        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss = self.sess.run(self.mean_loss)
        self.save_summary(summary_valid, train_step + self.conf.reload_step)
        if valid_loss < self.best_validation_loss:
            self.best_validation_loss = valid_loss
            improved_str = '(improved)'
            self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''
        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f}'
              .format(train_step, valid_loss, improved_str))
        print('-' * 60)

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        self.reload(step_num)
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='valid')
        self.num_test_batch = int(self.data_reader.y_test.shape[0] / self.conf.batch_size)
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_test_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.x: x_test, self.y: y_test, self.keep_prob: 1}
            self.sess.run(self.mean_loss_op, feed_dict=feed_dict)
        test_loss = self.sess.run(self.mean_loss)
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}'.format(test_loss))
        print('-' * 50)

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')
