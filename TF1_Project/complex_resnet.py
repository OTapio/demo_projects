from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import os 
#import tensorflow as tf 
import time
from datetime import datetime
import matplotlib.pyplot as plt


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import math
import errno
import shutil

import sys 
sys.path.append(os.path.dirname(os.getcwd()))


############### RESNET

print("\n PROGRAM BEGINS \n")

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):

  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)
  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

def sub_block(inputs, training, filters, kernel_size, strides, data_format, name):
  with tf.name_scope(name):
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    outputs = conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format)
  return outputs
	  




def myresnet(inputs, training, data_format):
  """Add operations to classify a batch of input images.

  Args:
    inputs: A Tensor representing a batch of input images.
    training: A boolean. Set to True to add operations required only when
      training the classifier.

  Returns:
    A logits Tensor with shape [<batch_size>, self.num_classes].
  """

  with tf.variable_scope('resnet_model', reuse=tf.AUTO_REUSE):
    data_format=data_format	  
    with tf.name_scope('Conv1'):

      channels = inputs.get_shape()[1].value
      kernel_size = 3
      filter_num = 64

      weights_1 = tf.Variable(tf.random_normal([kernel_size, kernel_size, channels, filter_num]))#,name="ZZZZZZZZZZZ")
      tf.add_to_collection('conv1_1_weights', weights_1)
      conv1_1 = tf.nn.conv2d(inputs, weights_1, strides=[1, 1, 1, 1], padding='SAME',data_format="NCHW")

       #self.conv1_1= tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3, strides=1,padding='SAME', 
      tf.add_to_collection('outputs_block_1', conv1_1)
    with tf.name_scope('Conv2-x'):
       
      shortcut2_1 = conv2d_fixed_padding(inputs=conv1_1, filters=128, kernel_size=1, strides=2,data_format=data_format)

      conv2_1 = sub_block(conv1_1,    training, filters=64,  kernel_size=3, strides=1, data_format=data_format, name='conv2-1')
      conv2_2 = sub_block(conv2_1,          training, filters=64,  kernel_size=3, strides=1, data_format=data_format, name='conv2-2')
      conv2_3 = sub_block(conv2_2,          training, filters=128, kernel_size=1, strides=2, data_format=data_format, name='conv2-3')

      shortcut2_2 = shortcut2_1 + conv2_3

      conv2_4 = sub_block(shortcut2_2,      training, filters=64,  kernel_size=3, strides=1, data_format=data_format, name='conv2-4')
      conv2_5 = sub_block(conv2_4,          training, filters=64,  kernel_size=3, strides=1, data_format=data_format, name='conv2-5')
      conv2_6 = sub_block(conv2_5,          training, filters=128, kernel_size=1, strides=1, data_format=data_format, name='conv2-6')
		
      outputs_block_1 = conv2_6 + shortcut2_2

    with tf.name_scope('Conv3-x'):

      shortcut3_1 = conv2d_fixed_padding(inputs=outputs_block_1, filters=128, kernel_size=1, strides=2, data_format=data_format)

      conv3_1 = sub_block(outputs_block_1,  training, filters=64,  kernel_size=1, strides=1, data_format=data_format, name='conv3-1')		
      conv3_2 = sub_block(conv3_1,          training, filters=128, kernel_size=3, strides=2, data_format=data_format, name='conv3-2')
      conv3_3 = sub_block(conv3_2,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv3-3')
		
      shortcut3_2 = shortcut3_1 + conv3_3

      conv3_4 = sub_block(shortcut3_2,      training, filters=64,  kernel_size=1, strides=1, data_format=data_format, name='conv3-4')
      conv3_5 = sub_block(conv3_4,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv3-5')
      conv3_6 = sub_block(conv3_5,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv3-6')

      shortcut3_3 = shortcut3_2 + conv3_6
 
      conv3_7 = sub_block(shortcut3_3,      training, filters=64,  kernel_size=1, strides=1, data_format=data_format, name='conv3-7')
      conv3_8 = sub_block(conv3_7,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv3-8')
      conv3_9 = sub_block(conv3_8,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv3-9')

      outputs_block_2 = shortcut3_3 + conv3_9
		
    with tf.name_scope('Conv4-x'):

      shortcut4_1 = outputs_block_2

      conv4_1 = sub_block(outputs_block_2,  training, filters=128, kernel_size=1, strides=1, data_format=data_format, name='conv4-1')
      conv4_2 = sub_block(conv4_1,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv4-2')
      conv4_3 = sub_block(conv4_2,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv4-3')
		
      shortcut4_2 = shortcut4_1 + conv4_2

      conv4_4 = sub_block(shortcut4_2,      training, filters=128, kernel_size=1, strides=1, data_format=data_format, name='conv4-4') 
      conv4_5 = sub_block(conv4_4,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv4-5')
      conv4_6 = sub_block(conv4_5,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv4-6')

      shortcut4_3 = shortcut4_2 + conv4_6

      conv4_7 = sub_block(shortcut4_3,      training, filters=128, kernel_size=1, strides=1, data_format=data_format, name='conv4-7') 
      conv4_8 = sub_block(conv4_7,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv4-8')
      conv4_9 = sub_block(conv4_8,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv4-9')

      shortcut4_4 = shortcut4_3 + conv4_9

      conv4_10 = sub_block(shortcut4_4,     training, filters=128, kernel_size=1, strides=1, data_format=data_format, name='conv4-10')
      conv4_11 = sub_block(conv4_10,        training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv4-11')
      conv4_12 = sub_block(conv4_11,        training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv4-12')

      outputs_block_3 = shortcut4_4 + conv4_12
		
    with tf.name_scope('Conv5-x'):

      shortcut5_1 = conv2d_fixed_padding(inputs=outputs_block_3, filters=512, kernel_size=1, strides=2,data_format=data_format)

      conv5_1 = sub_block(outputs_block_3,  training, filters=512, kernel_size=3, strides=2, data_format=data_format, name='conv5-1')
      conv5_2 = sub_block(conv5_1,          training, filters=512, kernel_size=3, strides=1, data_format=data_format, name='conv5-2')		
		
      shortcut5_2 = shortcut5_1 + conv5_2

      conv5_3 = sub_block(shortcut5_2,      training, filters=512, kernel_size=3, strides=1, data_format=data_format, name='conv5-3')
      conv5_4 = sub_block(conv5_3,          training, filters=512, kernel_size=3, strides=1, data_format=data_format, name='conv5-4')

      outputs_block_4 = shortcut5_2 + conv5_4

      tf.add_to_collection('outputs_block_4', outputs_block_4)

    inputs = batch_norm(outputs_block_4, training, data_format)
    inputs = tf.nn.relu(inputs)

    axes = [2, 3] if data_format == 'channels_first' else [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')
 
    inputs = tf.squeeze(inputs, axes)
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

############### CONVIZ
def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

		
def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]

        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')
############### UTILS

def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
      "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  images = parsed_features["image"]
  images = tf.decode_raw(images, tf.uint8)
  # channel first
  images = tf.reshape(images, [3, 32, 32])
  images = tf.cast(images, tf.float32)
  images = (images - 127) / 128.0 * 4
  return images, parsed_features["label"]

def get_data(data_dir, mode, batch_size):
  if mode == 'train':
    file = 'train.tfrecords'
  elif mode == 'validation':
    file = 'validation.tfrecords'
  elif mode == 'eval':
    file = 'eval.tfrecords'
  else:
    raise ValueError('mode should be %s or %s or %s' % ('train', 'validation', 'eval'))

  path = os.path.join(data_dir, file)
  dataset = tf.data.TFRecordDataset(path)
  dataset = dataset.map(_parse_function)

  if mode == 'train':
    dataset = dataset.shuffle(buffer_size=10000)

  dataset = dataset.repeat() 
  dataset = dataset.batch(batch_size) 
  itr = dataset.make_one_shot_iterator()
  images, labels = itr.get_next()
  return images, labels

def configure_learning_rate(global_step, num_samples, FLAGS):
    decay_steps = int(num_samples * FLAGS.num_epochs_per_decay / FLAGS.batch_size)

    return tf.train.exponential_decay(FLAGS.learning_rate,
            global_step,
            decay_steps,
            FLAGS.learning_rate_decay_factor,
            staircase=True,
            name='exponential_decay_learning_rate')

def get_cross_entropy(logits, labels):
  logits = tf.cast(logits, tf.float32)
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
  return cross_entropy

def get_accuracy(logits, labels):
  logits = tf.cast(logits, tf.float32)
  accuracy = tf.metrics.accuracy(labels, tf.argmax(logits, axis=1))
  return accuracy[1]

def get_reg_loss(weight_decay):
  reg_loss = weight_decay * tf.add_n(
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
  return reg_loss

def validate(sess, accuracy_val, batch_size, val_samples):
  num = 1
  while True:
    acc_value = sess.run(accuracy_val)
    num += batch_size
    print('Calculating accuracy on validation set: processed %d samples' % num, end='\r')
    if num > val_samples:
      return acc_value


############### MODEL_TRAIN


def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print ('Warning: {}'.format(e))


def create_dir(path):
    """
    Creates a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """
    if not os.path.exists(path):
        create_dir(path)

    if empty:
        empty_dir(path)


TRAIN_SAMPLES = 50000 
VAL_SAMPLES = 10000

##############################
# Flags most related to you #
##############################


tf.app.flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'epoch_number', 10,
    'Number of epoches')

tf.app.flags.DEFINE_string(
    'data_dir', None,
    'Directory of dataset.')

tf.app.flags.DEFINE_string(
    'train_dir', None,
    'Directory where checkpoints and event logs are written to.')



##############################
# Flags for learning rate #
##############################
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum for MomentumOptimizer.')

tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

##############################
# Flags for log and summary #
##############################
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 30,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'summary_every_n_steps', 30,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 300,
    'The frequency with which the model is saved, in seconds.')

FLAGS = tf.app.flags.FLAGS



##############################
#       Build ResNet         #
##############################
images, labels = get_data(FLAGS.data_dir, 'train', FLAGS.batch_size)
PLOT_DIR = FLAGS.train_dir + '/plots' 
print("\n\n IMAGES SHAPE ", images)

#resnet = myresnet(
#    num_classes=10, num_filters=64, kernel_size=3, conv_stride=1,
#    first_pool_size=None, first_pool_stride=None,
#   block_sizes=[2, 2, 2, 2], block_strides=[1, 2, 2, 2],
#    data_format='channels_first')

############################################
# Loss, Accuracy, Train, Summary and Saver #
############################################
weight_decay = 2e-4
num_classes = 10

logits = myresnet(images, training=True,data_format='channels_first')

cross_entropy = get_cross_entropy(logits, labels)
accuracy = get_accuracy(logits, labels)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

reg_loss = get_reg_loss(weight_decay)
tf.summary.scalar('reg_loss', reg_loss)

total_loss = cross_entropy + reg_loss
tf.summary.scalar('total_loss', total_loss)

global_step = tf.train.create_global_step()
learning_rate = configure_learning_rate(global_step, TRAIN_SAMPLES, FLAGS)
tf.summary.scalar('learning_rate', learning_rate)

optimizer = tf.train.MomentumOptimizer(
    learning_rate=learning_rate,
    momentum=FLAGS.momentum)
grads = optimizer.compute_gradients(total_loss)
train_op = optimizer.apply_gradients(grads, global_step=global_step)
summary_op = tf.summary.merge_all()

saver = tf.train.Saver(tf.trainable_variables())

############################################
#           For   validation               #
############################################
var_exclude = [v.name for v in tf.local_variables()]
images_val, labels_val = get_data(FLAGS.data_dir, 'validation', FLAGS.batch_size)
logits_val = myresnet(images_val, training=False,data_format='channels_first')
accuracy_val = get_accuracy(logits_val, labels_val)

# clear former accuracy information for validation
var_to_refresh = [v for v in tf.local_variables() if v.name not in var_exclude]
init_local_val = tf.variables_initializer(var_to_refresh)

#### HYPER PARAMETERS

print("\nHyper parameters: ")
print("TRAIN_SAMPLES: ", TRAIN_SAMPLES) 
print("VAL_SAMPLES: ", VAL_SAMPLES)
print("batch_size: ", FLAGS.batch_size)
print("epoch_number: ", FLAGS.epoch_number)
print("data_dir: ", FLAGS.data_dir)
print("train_dir: ", FLAGS.train_dir)
print("momentum: ", FLAGS.momentum)
print("learning_rate: ", FLAGS.learning_rate)
print("learning_rate_decay_factor: ", FLAGS.learning_rate_decay_factor)
print("num_epochs_per_decay: ", FLAGS.num_epochs_per_decay)
print("log_every_n_steps: ", FLAGS.log_every_n_steps)
print("summary_every_n_steps: ", FLAGS.summary_every_n_steps)
print("save_interval_secs: ", FLAGS.save_interval_secs)
print("\n")


x1 = tf.placeholder(tf.float32, [None, 3*32*32])
x2 = tf.placeholder(tf.float32, [None, 3072])

sess = tf.Session()


init_global = tf.global_variables_initializer() 
init_local = tf.local_variables_initializer()
train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/log', sess.graph)

# update trainable variables in the graph
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.group(train_op, update_ops)

sess.run(init_global)
sess.run(init_local)

x = tf.placeholder(tf.float32, [None, 32*32])
############################################
#           Let's start running            #
############################################
epoch_steps = int(TRAIN_SAMPLES / FLAGS.batch_size)
print('number of steps each epoch: ', epoch_steps)
epoch_index = 0
max_steps = FLAGS.epoch_number * epoch_steps
ori_time = time.time()
next_save_time = FLAGS.save_interval_secs

###########
path_2 = os.path.join(FLAGS.data_dir, 'train.tfrecords')
dataset_2 = tf.data.TFRecordDataset(path_2)
dataset_2 = dataset_2.map(_parse_function)
dataset_2 = dataset_2.batch(50000)
whole_dataset_tensors = tf.data.experimental.get_single_element(dataset_2)#tf.contrib.data.get_single_element(dataset)
images_2 = sess.run(whole_dataset_tensors)
images_3 = np.reshape(images_2[0], [-1, 3072])

##########
for step in range(max_steps):
    start_time = time.time()
    if step % epoch_steps == 0:
        epoch_index += 1
        if epoch_index > 0:
          sess.run(init_local_val)
          accuracy_val_value = validate(sess, accuracy_val, FLAGS.batch_size, VAL_SAMPLES)
          duration = time.time() - start_time
          duration = float(duration) / 60.0
          val_format = 'Time of validation after epoch %02d: %.2f mins, val accuracy: %.4f'
          print(val_format % (epoch_index - 1, duration, accuracy_val_value))
          

    [_, total_l_value, entropy_l_value, reg_l_value, acc_value] = \
        sess.run([train_op, total_loss, cross_entropy, reg_loss, accuracy])

    total_duration = time.time() - ori_time
    total_duration = float(total_duration)

    assert not np.isnan(total_l_value), 'Model diverged with loss = NaN' 

    if step % FLAGS.log_every_n_steps == 0:
      format_str = ('Epoch %02d/%2d time=%.2f mins: step %d total loss=%.4f loss=%.4f reg loss=%.4f accuracy=%.4f')
      print(format_str % (epoch_index, FLAGS.epoch_number, total_duration / 60.0, step, total_l_value, entropy_l_value, reg_l_value, acc_value))

    if step % FLAGS.summary_every_n_steps == 0:
      summary_str = sess.run(summary_op)
      train_writer.add_summary(summary_str, step)

    if total_duration > next_save_time:
      next_save_time += FLAGS.save_interval_secs
      checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      save_path = saver.save(sess, checkpoint_path, global_step=global_step)
      print('saved model to %s' % save_path)

outputs_block_1 = sess.run([tf.get_collection('outputs_block_1')], feed_dict={x1: images_3})
for i, c in enumerate(outputs_block_1[0]):
  plot_conv_output(c, 'outputs_block_1{}'.format(i))
	
outputs_block_4 = sess.run([tf.get_collection('outputs_block_4')], feed_dict={x2: images_3})
for i, c in enumerate(outputs_block_4[0]):
  plot_conv_output(c, 'outputs_block_4{}'.format(i))

checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
save_path = saver.save(sess, checkpoint_path, global_step=global_step)
print('saved the final model to %s' % save_path)

sess.run(init_local_val)
accuracy_val_value = validate(sess, accuracy_val, FLAGS.batch_size, VAL_SAMPLES)
print("\n")
conv1_1_weights = sess.run([tf.get_collection('conv1_1_weights')])
for i, c in enumerate(conv1_1_weights[0]):
  plot_conv_weights(c, 'conv1_1_weights{}'.format(i))



	
print("\n PROGRAM ENDS \n")

# Time of validation after epoch 14: 0.06 mins, val accuracy: 0.0867
# Epoch 15/15 time=14.32 mins: step 10950 total loss=0.5885 loss=0.1103 reg loss=0.4782 accuracy=0.8141
# Epoch 15/15 time=14.35 mins: step 10980 total loss=0.8170 loss=0.3395 reg loss=0.4775 accuracy=0.8145
# Epoch 15/15 time=14.39 mins: step 11010 total loss=0.6088 loss=0.1321 reg loss=0.4767 accuracy=0.8148
# Epoch 15/15 time=14.43 mins: step 11040 total loss=0.5561 loss=0.0804 reg loss=0.4758 accuracy=0.8152
# Epoch 15/15 time=14.46 mins: step 11070 total loss=0.6025 loss=0.1277 reg loss=0.4748 accuracy=0.8156
# Epoch 15/15 time=14.50 mins: step 11100 total loss=0.5824 loss=0.1084 reg loss=0.4739 accuracy=0.8160
# Epoch 15/15 time=14.54 mins: step 11130 total loss=0.6010 loss=0.1275 reg loss=0.4735 accuracy=0.8163
# Epoch 15/15 time=14.57 mins: step 11160 total loss=0.6221 loss=0.1487 reg loss=0.4734 accuracy=0.8166
# Epoch 15/15 time=14.61 mins: step 11190 total loss=0.6207 loss=0.1472 reg loss=0.4735 accuracy=0.8170
# Epoch 15/15 time=14.65 mins: step 11220 total loss=0.6315 loss=0.1587 reg loss=0.4728 accuracy=0.8173
# Epoch 15/15 time=14.68 mins: step 11250 total loss=0.6162 loss=0.1435 reg loss=0.4726 accuracy=0.8176
# Epoch 15/15 time=14.72 mins: step 11280 total loss=0.7138 loss=0.2416 reg loss=0.4722 accuracy=0.8179
# Epoch 15/15 time=14.75 mins: step 11310 total loss=0.6195 loss=0.1476 reg loss=0.4719 accuracy=0.8183
# Epoch 15/15 time=14.79 mins: step 11340 total loss=0.6090 loss=0.1375 reg loss=0.4715 accuracy=0.8186
# Epoch 15/15 time=14.83 mins: step 11370 total loss=0.5912 loss=0.1203 reg loss=0.4709 accuracy=0.8189
# Epoch 15/15 time=14.86 mins: step 11400 total loss=0.7781 loss=0.3074 reg loss=0.4708 accuracy=0.8192
# Epoch 15/15 time=14.90 mins: step 11430 total loss=0.6834 loss=0.2122 reg loss=0.4712 accuracy=0.8195
# Epoch 15/15 time=14.93 mins: step 11460 total loss=0.7283 loss=0.2567 reg loss=0.4717 accuracy=0.8198
# Epoch 15/15 time=14.97 mins: step 11490 total loss=0.7359 loss=0.2636 reg loss=0.4723 accuracy=0.8201
# saved model to testi_complex_1\model.ckpt-11511
# Epoch 15/15 time=15.03 mins: step 11520 total loss=0.6181 loss=0.1455 reg loss=0.4726 accuracy=0.8204
# Epoch 15/15 time=15.07 mins: step 11550 total loss=0.5905 loss=0.1184 reg loss=0.4721 accuracy=0.8207
# Epoch 15/15 time=15.10 mins: step 11580 total loss=0.5676 loss=0.0959 reg loss=0.4717 accuracy=0.8210
# Epoch 15/15 time=15.14 mins: step 11610 total loss=0.5674 loss=0.0957 reg loss=0.4718 accuracy=0.8213
# Epoch 15/15 time=15.17 mins: step 11640 total loss=0.7247 loss=0.2530 reg loss=0.4717 accuracy=0.8216
# Epoch 15/15 time=15.21 mins: step 11670 total loss=0.5975 loss=0.1255 reg loss=0.4720 accuracy=0.8219
# Epoch 15/15 time=15.25 mins: step 11700 total loss=0.6110 loss=0.1393 reg loss=0.4717 accuracy=0.8222
# conv_output
# conv_output
# conv_output
# conv_output
# saved the final model to testi_complex_1\model.ckpt-11715
# Calculating accuracy on validation set: processed 10049 samples

# conv_weights
# conv_weights

 # OHJELMA PAATTYY
