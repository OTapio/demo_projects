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
starttime = time.time()
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
	  
class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, num_classes, num_filters, kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               data_format=None):

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    with tf.variable_scope('resnet_model', reuse=tf.AUTO_REUSE):
      data_format=self.data_format	  
      with tf.name_scope('Conv1'):
#        print("Conv1 input:  ",inputs.shape)
        self.output_Conv1 = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=3, strides=1, data_format=data_format)
#        print("Conv1 output: ",self.output_Conv1.shape)
      with tf.name_scope('Conv2-x'):
        
        self.shortcut2_1 = conv2d_fixed_padding(inputs=self.output_Conv1, filters=64, kernel_size=1, strides=1,data_format=data_format)
#        print("conv2-1 padding: ",self.shortcut2_1.shape)
        self.conv2_1 = sub_block(self.output_Conv1,         training, filters=64,  kernel_size=3, strides=1, data_format=data_format, name='conv2-1')
#        print("conv2_1 : ",self.conv2_1.shape)
        self.conv2_2 = sub_block(self.conv2_1,              training, filters=64,  kernel_size=3, strides=1, data_format=data_format, name='conv2-2')
 #       print("conv2_2 : ",self.conv2_2.shape)
        self.shortcut2_2 = self.shortcut2_1 + self.conv2_2
        #print(self.shortcut2_2.shape)
        self.conv2_3 = sub_block(self.shortcut2_2,          training, filters=64,  kernel_size=3, strides=1, data_format=data_format, name='conv2-3')
#        print("conv2_3 : ",self.conv2_3.shape)
        self.conv2_4 = sub_block(self.conv2_3,              training, filters=64,  kernel_size=3, strides=1, data_format=data_format, name='conv2-4')
#        print("conv2_4 : ",self.conv2_4.shape)
        self.outputs_block_1 = self.conv2_4 + self.shortcut2_2
        #print(self.outputs_block_1.shape)
      with tf.name_scope('Conv3-x'):

        self.shortcut3_1 = conv2d_fixed_padding(inputs=self.outputs_block_1, filters=128, kernel_size=1, strides=2, data_format=data_format)
#        print("conv3 padding: ",self.shortcut3_1.shape)		  
        self.conv3_1 = sub_block(self.outputs_block_1,  training, filters=128, kernel_size=3, strides=2, data_format=data_format, name='conv3-1')		
#        print("conv3_1 : ",self.conv3_1.shape)
        self.conv3_2 = sub_block(self.conv3_1,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv3-2')
#        print("conv3_2 : ",self.conv3_2.shape)
        self.shortcut3_2 = self.shortcut3_1 + self.conv3_2
        #print(self.shortcut3_2.shape)
        self.conv3_3 = sub_block(self.shortcut3_2,          training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv3-3')
#        print("conv3_3 : ",self.conv3_3.shape)
        self.conv3_4 = sub_block(self.conv3_3,              training, filters=128, kernel_size=3, strides=1, data_format=data_format, name='conv3-4')
#        print("conv3_4 : ",self.conv3_4.shape)
        self.outputs_block_2 = self.conv3_4 + self.shortcut3_2
        #print(self.outputs_block_2.shape)
      with tf.name_scope('Conv4-x'):

        self.shortcut4_1 = conv2d_fixed_padding(inputs=self.outputs_block_2, filters=256, kernel_size=1, strides=2, data_format=data_format)	
#        print("conv4 padding: ",self.shortcut4_1.shape)
        self.conv4_1 = sub_block(self.outputs_block_2, training, filters=256, kernel_size=3, strides=2, data_format=data_format, name='conv4-1')
#        print("conv4_1 : ",self.conv4_1.shape)
        self.conv4_2 = sub_block(self.conv4_1,          training, filters=256, kernel_size=3, strides=1, data_format=data_format, name='conv4-2')
#        print("conv4_2 : ",self.conv4_2.shape)
        self.shortcut4_2 = self.shortcut4_1 + self.conv4_2
        #print(self.shortcut4_2.shape)
        self.conv4_3 = sub_block(self.shortcut4_2,          training, filters=256, kernel_size=3, strides=1, data_format=data_format, name='conv4-3') 
#        print("conv4_3 : ",self.conv4_3.shape)
        self.conv4_4 = sub_block(self.conv4_3,          training, filters=256, kernel_size=3, strides=1, data_format=data_format, name='conv4-4')
#        print("conv4_4 : ",self.conv4_4.shape)
        self.outputs_block_3 = self.conv4_4 + self.shortcut4_2
        #print(self.outputs_block_3.shape)
      with tf.name_scope('Conv5-x'):

        self.shortcut5_1 = conv2d_fixed_padding(inputs=self.outputs_block_3, filters=512, kernel_size=1, strides=2,data_format=data_format)		
#        print("conv5 padding: ",self.shortcut5_1.shape)
        self.conv5_1 = sub_block(self.outputs_block_3, training, filters=512, kernel_size=3, strides=2, data_format=data_format, name='conv5-1')
#        print("conv5_1 : ",self.conv5_1.shape)
        self.conv5_2 = sub_block(self.conv5_1,          training, filters=512, kernel_size=3, strides=1, data_format=data_format, name='conv5-2')		
#        print("conv5_2 : ",self.conv5_2.shape)
        self.shortcut5_2 = self.shortcut5_1 + self.conv5_2
        #print(self.shortcut5_2.shape)
        self.conv5_3 = sub_block(self.shortcut5_2,          training, filters=512, kernel_size=3, strides=1, data_format=data_format, name='conv5-3')
#        print("conv5_3 : ",self.conv5_3.shape)
        self.conv5_4 = sub_block(self.conv5_3,          training, filters=512, kernel_size=3, strides=1, data_format=data_format, name='conv5-4')
#        print("conv5_4 : ",self.conv5_4.shape)
        self.outputs_block_4 = self.conv5_4  + self.shortcut5_2	  
        #print(self.outputs_block_4.shape)
      inputs = batch_norm(self.outputs_block_4, training, self.data_format)
#      print(inputs.shape)
      inputs = tf.nn.relu(inputs)
#      print(inputs.shape)
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
#      print(inputs.shape)
      inputs = tf.identity(inputs, 'final_reduce_mean')
#      print(inputs.shape)
      inputs = tf.squeeze(inputs, axes)
#      print(inputs.shape)
      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
#      print(inputs.shape)
      inputs = tf.identity(inputs, 'final_dense')
#      print(inputs.shape)
      return inputs



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

resnet = Model(
    num_classes=10, num_filters=64, kernel_size=3, conv_stride=1,
    first_pool_size=None, first_pool_stride=None,
    data_format='channels_first')

############################################
# Loss, Accuracy, Train, Summary and Saver #
############################################
weight_decay = 2e-4

logits = resnet(images, training=True)

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
logits_val = resnet(images_val, training=False)
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
sess = tf.Session()


init_global = tf.global_variables_initializer() 
init_local = tf.local_variables_initializer()
train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/log', sess.graph)

# update trainable variables in the graph
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.group(train_op, update_ops)

sess.run(init_global)
sess.run(init_local)

############################################
#           Let's start running            #
############################################
epoch_steps = int(TRAIN_SAMPLES / FLAGS.batch_size)
print('number of steps each epoch: ', epoch_steps)
epoch_index = 0
max_steps = FLAGS.epoch_number * epoch_steps
ori_time = time.time()
next_save_time = FLAGS.save_interval_secs


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

checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
save_path = saver.save(sess, checkpoint_path, global_step=global_step)
print('saved the final model to %s' % save_path)

sess.run(init_local_val)
accuracy_val_value = validate(sess, accuracy_val, FLAGS.batch_size, VAL_SAMPLES)

print("accuracy_val_value: ", accuracy_val_value)
print("\n")
endtime = time.time()
print("\nTime: ",endtime - starttime)
	
print("\n PROGRAM ENDS \n")


# Time of validation after epoch 14: 0.07 mins, val accuracy: 0.7847
# Epoch 15/15 time=17.49 mins: step 10950 total loss=0.5596 loss=0.1032 reg loss=0.4564 accuracy=0.8354
# Epoch 15/15 time=17.54 mins: step 10980 total loss=0.5243 loss=0.0687 reg loss=0.4556 accuracy=0.8357
# Epoch 15/15 time=17.58 mins: step 11010 total loss=0.5165 loss=0.0616 reg loss=0.4549 accuracy=0.8361
# Epoch 15/15 time=17.63 mins: step 11040 total loss=0.6055 loss=0.1517 reg loss=0.4539 accuracy=0.8364
# Epoch 15/15 time=17.67 mins: step 11070 total loss=0.5989 loss=0.1455 reg loss=0.4534 accuracy=0.8367
# Epoch 15/15 time=17.72 mins: step 11100 total loss=0.5310 loss=0.0780 reg loss=0.4530 accuracy=0.8371
# Epoch 15/15 time=17.76 mins: step 11130 total loss=0.5101 loss=0.0573 reg loss=0.4527 accuracy=0.8374
# Epoch 15/15 time=17.81 mins: step 11160 total loss=0.5274 loss=0.0752 reg loss=0.4523 accuracy=0.8377
# Epoch 15/15 time=17.85 mins: step 11190 total loss=0.6009 loss=0.1489 reg loss=0.4520 accuracy=0.8380
# Epoch 15/15 time=17.89 mins: step 11220 total loss=0.5793 loss=0.1273 reg loss=0.4520 accuracy=0.8383
# Epoch 15/15 time=17.94 mins: step 11250 total loss=0.5895 loss=0.1372 reg loss=0.4523 accuracy=0.8386
# Epoch 15/15 time=17.98 mins: step 11280 total loss=0.8106 loss=0.3584 reg loss=0.4522 accuracy=0.8389
# Epoch 15/15 time=18.03 mins: step 11310 total loss=0.6082 loss=0.1561 reg loss=0.4521 accuracy=0.8392
# Epoch 15/15 time=18.07 mins: step 11340 total loss=0.5757 loss=0.1239 reg loss=0.4518 accuracy=0.8395
# Epoch 15/15 time=18.11 mins: step 11370 total loss=0.5238 loss=0.0723 reg loss=0.4515 accuracy=0.8398
# Epoch 15/15 time=18.15 mins: step 11400 total loss=0.7250 loss=0.2735 reg loss=0.4515 accuracy=0.8401
# Epoch 15/15 time=18.20 mins: step 11430 total loss=0.5691 loss=0.1169 reg loss=0.4521 accuracy=0.8404
# Epoch 15/15 time=18.24 mins: step 11460 total loss=0.5734 loss=0.1213 reg loss=0.4521 accuracy=0.8407
# Epoch 15/15 time=18.28 mins: step 11490 total loss=0.6202 loss=0.1681 reg loss=0.4522 accuracy=0.8409
# Epoch 15/15 time=18.34 mins: step 11520 total loss=0.5869 loss=0.1354 reg loss=0.4515 accuracy=0.8412
# Epoch 15/15 time=18.38 mins: step 11550 total loss=0.5337 loss=0.0830 reg loss=0.4507 accuracy=0.8415
# Epoch 15/15 time=18.42 mins: step 11580 total loss=0.5174 loss=0.0671 reg loss=0.4502 accuracy=0.8418
# Epoch 15/15 time=18.47 mins: step 11610 total loss=0.5217 loss=0.0718 reg loss=0.4499 accuracy=0.8422
# Epoch 15/15 time=18.51 mins: step 11640 total loss=0.5087 loss=0.0591 reg loss=0.4496 accuracy=0.8424
# Epoch 15/15 time=18.56 mins: step 11670 total loss=0.5671 loss=0.1181 reg loss=0.4490 accuracy=0.8428
# Epoch 15/15 time=18.60 mins: step 11700 total loss=0.4862 loss=0.0375 reg loss=0.4487 accuracy=0.8430
# saved the final model to testi_1_Final\model.ckpt-11715
# accuracy_val_value:  0.75865847ion set: processed 10049 samples