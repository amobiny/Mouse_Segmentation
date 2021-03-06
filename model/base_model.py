import tensorflow as tf
from Data_Loader import DataLoader
import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import scipy
import os.path as path
from PIL import Image
from skimage import img_as_ubyte

# add a line for pull test
class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.is_training = True
        self.input_shape = [None, None, None, self.conf.in_channel]
        self.output_shape = [None, None, None, self.conf.out_channel]
        self.output_shape = [None, None, None, self.conf.out_channel]
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
                    #norm_factor = tf.shape(self.y)[1] * tf.shape(self.y)[2]
                    #self.loss = tf.norm(self.y - self.y_pred)
                    self.loss=tf.losses.mean_squared_error(self.y , self.y_pred)#
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

        if self.idxs is not None:                   # if we have a mask for training, idxs is the list of pixels used for training. 1 in the mask
            self.data_reader.idxs= self.idxs

        # self.data_reader.load_crop_data(mode='valid')
        self.data_reader.get_data(mode='valid')
        self.num_val_batch = max (int(self.data_reader.y_valid.shape[0] / self.conf.batch_size),1)
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
                self.evaluate(train_step)

    def evaluate(self, train_step):
        self.is_training = False
        self.sess.run(tf.local_variables_initializer())
        # pred_mask = np.zeros_like(self.data_reader.y_valid)
        # for step in range(self.num_val_batch):
        #     start = step * self.conf.batch_size
        #     end = (step + 1) * self.conf.batch_size
        #     x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
        feed_dict = {self.x: self.data_reader.x_valid, self.y: self.data_reader.y_valid, self.keep_prob: 1}
        pred_mask,  _ = self.sess.run([self.y_pred, self.mean_loss_op], feed_dict=feed_dict)
        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss = self.sess.run(self.mean_loss)
        self.save_summary(summary_valid, train_step + self.conf.reload_step)
        self.save_fig(train_step,
                      pred_mask.reshape(-1, self.data_reader.val_img_size[0], self.data_reader.val_img_size[1], self.conf.out_channel),
                      self.data_reader.y_valid.reshape(-1, self.data_reader.val_img_size[0], self.data_reader.val_img_size[1], self.conf.out_channel),
                      valid_loss, mode='valid')
        if valid_loss < self.best_validation_loss:
            self.best_validation_loss = valid_loss
            improved_str = '(improved)'
            self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''
        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f} {2}'
              .format(train_step, valid_loss, improved_str))
        print('-' * 60)

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        self.reload(step_num)
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='test')
        self.num_test_batch = max (int(self.data_reader.y_test.shape[0] / self.conf.batch_size) , 1)
        pred_mask = np.zeros_like(self.data_reader.y_test)
        self.is_training = False
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_test_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.x: x_test, self.y: y_test, self.keep_prob: 1}
            pred_mask[start:end], _ = self.sess.run([self.y_pred, self.mean_loss_op], feed_dict=feed_dict)
        test_loss = self.sess.run(self.mean_loss)
        self.save_fig(step_num, pred_mask, self.data_reader.y_test, test_loss)
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}'.format(test_loss))
        print('-' * 50)

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir+self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir+self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model at step# {}'.format(step))
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')

    def save_fig(self, step, pred, y, loss, mode='test'):
        num_images = pred.shape[0]
        fig, axs = plt.subplots(nrows=num_images, ncols=3)
        axs = axs.reshape(-1, 3)
        for ii in range(num_images):
            pred_mask = pred[ii]
            true_mask = y[ii]
            ax = axs[ii, 0]
            ax.imshow(true_mask)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Groung Truth')
            ax = axs[ii, 1]
            ax.imshow(pred_mask)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Prediction')
            ax = axs[ii, 2]
            ax.imshow(np.abs(true_mask-pred_mask))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Difference)')
            ax.set_xlabel('loss='+str(loss))
            fig.set_size_inches(6, 2*num_images)
            fig.savefig(self.conf.modeldir+self.conf.run_name+'/'+mode+'_step_{0}_image{1}.png'.format(step, ii))
            misc.imsave(self.conf.modeldir+self.conf.run_name+'/'+mode+'_step_{0}_image{1}.bmp'.format(step, ii),pred_mask)
        plt.close('all')

    def load_mask(self):
        # get file names in masks directory sorted in alphabetical order
        mask_path = self.conf.data_dir +'mask/'
        file_list =[]
        mask_files=[]
        for f in os.listdir(mask_path):
            file_list.append(f)
        file_list.sort()
        #check if they are really file :)
        for f in file_list:
            if not f.startswith('.'):
                file_name = path.join(mask_path, f)
                if path.isfile(file_name) and not (file_name in mask_files):
                    mask_files.append(file_name)

        if len(mask_files) < 1:
            raise ValueError('Missing masks files in folder %s!' % mask_path)

        num_mask = len(mask_files)
        idx =[]
        for i in range(num_mask):
            mask = scipy.misc.imread(mask_files[i])
            idx_temp = np.transpose(np.nonzero(mask[:, :, 1])) # non_zero indices for each mask
            max_bottom_left_front_corner = (np.max(idx_temp[:, 0]) - self.conf.height - 1, np.max(idx_temp[:, 1]) - self.conf.width - 1)
            for j in idx_temp:                #check the nonzero points of masks and exclude border points
                if j[0] < max_bottom_left_front_corner[0] and j[1] < max_bottom_left_front_corner[1]:
                    idx.append(j)

        #mask_path = self.conf.data_dir + 'tr_mask.bmp'
        #mask = scipy.misc.imread(mask_path)
        #idx = np.transpose(np.nonzero(mask[:, :, 1]))
        #idxs = []  # indices close to borders which cannot be bottom_left corner are excluded from idx
        #max_bottom_left_front_corner = (np.max(idx[:,0]) - self.conf.height - 1, np.max(idx[:,1]) - self.conf.width - 1)
        #max_bottom_left_front_corner = (self.conf.img_size[0] - self.conf.height - 1, self.conf.img_size[1] - self.conf.width - 1)
        #for i in idx:
        #    if i[0] < max_bottom_left_front_corner[0] and i[1] < max_bottom_left_front_corner[1]:
        #         idxs.append(i)

        self.idxs = np.asarray(idx)
