import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
flags.DEFINE_integer('step_num', 20000, 'Reload step to test the model')

# Training logs
flags.DEFINE_integer('max_step', 100000, '# of step for training')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 100, 'Number of step to evaluate the network on Validation data')
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# Hyper-parameters
flags.DEFINE_string('loss_type', 'MSE', 'Mean-Squared Error')
flags.DEFINE_float('lmbda', 1e-3, 'L2 regularization coefficient')
flags.DEFINE_integer('batch_size', 8, 'training batch size')
flags.DEFINE_integer('val_batch_size', 1, 'training batch size')

# data
flags.DEFINE_integer('num_tr', 1280, 'Total number of training images')
flags.DEFINE_string('train_data_dir', './data/train_data/', 'Training data directory')
flags.DEFINE_string('valid_data_dir', './data/valid_data/', 'Validation data directory')
flags.DEFINE_string('test_data_dir', './data/test_data/', 'Test data directory')
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 40, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('height', 64, 'Network input height size')
flags.DEFINE_integer('width', 64, 'Network input width size')
flags.DEFINE_integer('in_channel', 1, 'Number of input channels')
flags.DEFINE_integer('out_channel', 60, 'Number of output channels')

# Directories
flags.DEFINE_string('run_name', 'run01', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Model directory')


# network architecture
flags.DEFINE_boolean('use_BN', True, 'Adds Batch-Normalization to all convolutional layers')
flags.DEFINE_integer('start_channel_num', 16, 'start number of outputs for the first conv layer')
flags.DEFINE_integer('filter_size', 3, 'Filter size for the conv and deconv layers')
flags.DEFINE_integer('pool_filter_size', 2, 'Filter size for pooling layers')
flags.DEFINE_integer('drop_out_rate', 0.9, 'Dropout rate')

args = tf.app.flags.FLAGS
