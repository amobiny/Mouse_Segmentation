import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train or test')
flags.DEFINE_string('model', 'Tiramisu', 'FCN or Tiramisu or Densenet or CNN')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
flags.DEFINE_integer('step_num', 1500, 'Reload step to test the model')

# Training logs
flags.DEFINE_integer('max_step', 2000, '# of step for training')
flags.DEFINE_integer('SUMMARY_FREQ', 100, 'Number of step to save summary')
flags.DEFINE_integer('VAL_FREQ', 100, 'Number of step to evaluate the network on Validation data') #default = 1000
flags.DEFINE_float('init_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_min', 1e-5, 'Minimum learning rate')

# Hyper-parameters
flags.DEFINE_string('loss_type', 'MSE', 'Mean-Squared Error')
flags.DEFINE_boolean('use_reg', True, 'Adds L2 regularization')#default False
flags.DEFINE_float('lmbda', 1e-3, 'L2 regularization coefficient')
flags.DEFINE_integer('batch_size',4, 'training batch size')

# data
flags.DEFINE_integer('num_tr', 1, 'Total number of training images')
flags.DEFINE_string('data_dir', './data/', 'Data directory')
flags.DEFINE_boolean('normalize', True, 'Whether to load the normalized data or not') # True
flags.DEFINE_boolean('mask', True, 'Whether to use training mask or not') # True
flags.DEFINE_boolean('data_augment', True, 'Adds augmentation to data')
flags.DEFINE_integer('max_angle', 180, 'Maximum rotation angle along each axis; when applying augmentation')
flags.DEFINE_integer('img_size', [1024,512], 'Size of the large original image')
flags.DEFINE_integer('height', 32, 'Network input height size')
flags.DEFINE_integer('width', 32, 'Network input width size')
flags.DEFINE_integer('in_channel', 60, 'Number of input channels')
flags.DEFINE_integer('out_channel', 3, 'Number of output channels')

# Directories
flags.DEFINE_string('run_name', 'tira_575_kid_dapi', 'Run name')
flags.DEFINE_string('logdir', './Results/log_dir/', 'Logs directory')
flags.DEFINE_string('modeldir', './Results/model_dir/', 'Model directory')
flags.DEFINE_string('model_name', 'model', 'Nothing but a name :)')

# network architecture
flags.DEFINE_boolean('use_BN', True, 'Adds Batch-Normalization to all convolutional layers')# use to be True
flags.DEFINE_integer('start_channel_num', 16, 'start number of outputs for the first conv layer')
flags.DEFINE_integer('filter_size', 3, 'Filter size for the conv and deconv layers')
flags.DEFINE_integer('pool_filter_size', 2, 'Filter size for pooling layers')
flags.DEFINE_float('drop_out_rate', 0.8, 'Dropout rate') # used to be 0.8
flags.DEFINE_integer('growth_rate', 2, 'growth rate of the DenseNet')
flags.DEFINE_float('theta_down', 0.5, 'encoder compression rate')
flags.DEFINE_float('theta_up', 0.5, 'decoder compression rate')

args = tf.app.flags.FLAGS
