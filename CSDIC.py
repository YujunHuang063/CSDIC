# -*- coding: utf-8 -*-
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys
import pdb

from absl import app
from absl.flags import argparse_flags
import numpy as np
import cv2
import logging
import os
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow_compression.python.layers import parameterizers
#tf.enable_eager_execution()
import tensorflow_compression as tfc

from preprocess import *
from utils import *

BASE_NUM = 14
_NUM = 48
Y_NUM = 28
UV_NUM = 10
HY_NUM = 3
Y_HY_NUM = 18
UV_HY_NUM = 6
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

CITYSCAPE_DIR = '../Cityscapes_dataset' # cofig your data path


CITYSCAPE_IMG_DIR = os.path.join(CITYSCAPE_DIR, 'leftImg8bit')
CITYSCAPE_ANNO_DIR = os.path.join(CITYSCAPE_DIR, 'gtFine_trainvaltest/gtFine')

def read_png(filename1):
  """Loads a PNG image file."""
  string = tf.read_file(filename1)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)

class AverageMeter(object):
  def __init__(self, max_len):
    self.reset()
    self.max_len = max_len

  def reset(self):
    self.val = 0
    self.avg = 0
    self.record = []

  def update(self, val):
    self.record.append(val)
    if len(self.record)>self.max_len:
      del(self.record[0])
    self.val = val
    self.avg = np.sum(self.record) / len(self.record)

class AnalysisSynthesisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, depth, *args, **kwargs):
    self.num_filters = num_filters
    self.init_build = False
    self.depth = depth
    self.block_width = 10
    super(AnalysisSynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.y_layers1 = [
        tfc.SignalConv2D(
            Y_NUM, (self.block_width, self.block_width), name="layer1_0", corr=True, strides_down=self.block_width,
            padding="valid", use_bias=False,
            activation=None, dtype=self.dtype),
    ]
    self.u_layers1 = [
        tfc.SignalConv2D(
            UV_NUM, (self.block_width, self.block_width), name="layer1_1", corr=True, strides_down=self.block_width,
            padding="valid", use_bias=False,
            activation=None, dtype=self.dtype),
    ]
    self.v_layers1 = [
        tfc.SignalConv2D(
            UV_NUM, (self.block_width, self.block_width), name="layer1_2", corr=True, strides_down=self.block_width,
            padding="valid", use_bias=False,
            activation=None, dtype=self.dtype),
    ]
     
    self.rho = self.add_variable(
            "rho", dtype=self.dtype,
            shape=(self.depth+3,),
            initializer=tf.initializers.random_uniform(0.01, 1.0))
    
    self.yuv_layers2 = [
        tfc.SignalConv2D(
            3, (self.block_width, self.block_width), name="layer2_0", corr=False, strides_up=self.block_width,
            padding="same_zeros", use_bias=True,
            activation=None, dtype=self.dtype),
    ]

    
    self.deeprec_list = []
    num=0
    self.deeprec_list.append(tfc.SignalConv2D(self.num_filters, (3, 3), name="layer3_"+str(num), corr=False, padding="same_zeros", use_bias=True, activation=None, dtype=self.dtype))
    num+=1
    for i in range(self.depth):
        self.deeprec_list.append(tfc.SignalConv2D(self.num_filters, (3, 3), name="layer3_"+str(num), corr=False, padding="same_zeros", use_bias=True, activation=partial(tf.nn.leaky_relu, alpha=0.01)
            , dtype=self.dtype))
        num+=1
        self.deeprec_list.append(tfc.SignalConv2D(self.num_filters, (3, 3), name="layer3_"+str(num), corr=False, padding="same_zeros", use_bias=True, activation=None, dtype=self.dtype))
        num+=1
        self.deeprec_list.append(tfc.SignalConv2D(3, (3, 3), name="layer3_"+str(num), corr=False, padding="same_zeros", use_bias=True, activation=None, dtype=self.dtype))
        num+=1
        if i<self.depth-1:
            self.deeprec_list.append(tfc.SignalConv2D(self.num_filters, (3, 3), name="layer3_"+str(num), corr=False, padding="same_zeros", use_bias=True, activation=None, dtype=self.dtype))
            num+=1
    
    super(AnalysisSynthesisTransform, self).build(input_shape)

  def normflow(self, tensor):
    #paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    tensor = self.YUV_Measurement(tensor) 
    return tensor      

  def YUV_Measurement(self, tensor):
    for i, layer in enumerate(self.y_layers1):
      tensor1 = layer(tensor[:, :, :, :1])
      
    for i, layer in enumerate(self.u_layers1):
      tensor2 = layer(tensor[:, :, :, 1:2])

    for i, layer in enumerate(self.v_layers1):  
      tensor3 = layer(tensor[:, :, :, 2:])   
       
    tensor = tf.concat([tensor1, tensor2, tensor3], 3)  

    return tensor

  def YUV_linear_initializtion(self, tensor, _shape):
      
    for i, layer in enumerate(self.yuv_layers2):
      layer(tensor)
      tensor = tf.nn.conv2d_transpose(tensor, tf.transpose(layer._kernel, (0, 1, 3, 2)), output_shape=_shape, strides=[1, self.block_width, self.block_width, 1], padding="VALID")
      tensor = tf.nn.bias_add(tensor, layer._bias)

    return tensor

  def YUV_measurement_transpose(self, tensor, _shape):
    _shape[3]=1
      
    for i, layer in enumerate(self.y_layers1):
      tensor1 = tensor[:, :, :, :Y_NUM]
      tensor1 = tf.nn.conv2d_transpose(tensor1, layer._kernel, output_shape=_shape, strides=[1, self.block_width, self.block_width, 1], padding="VALID")

    for i, layer in enumerate(self.u_layers1):
      tensor2 = tensor[:, :, :, Y_NUM:Y_NUM+UV_NUM]
      tensor2 = tf.nn.conv2d_transpose(tensor2, layer._kernel, output_shape=_shape, strides=[1, self.block_width, self.block_width, 1], padding="VALID")

    for i, layer in enumerate(self.v_layers1):
      tensor3 = tensor[:, :, :, Y_NUM+UV_NUM:Y_NUM+2*UV_NUM]
      tensor3 = tf.nn.conv2d_transpose(tensor3, layer._kernel, output_shape=_shape, strides=[1, self.block_width, self.block_width, 1], padding="VALID")
      
    tensor = tf.concat([tensor1, tensor2, tensor3], 3) 
    return tensor
        
  def reverseflow(self, tensor):
    
    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
    tensor.set_shape([tensor.shape[0], tensor.shape[1], tensor.shape[2], _NUM])
    _shape = [self.shape[0][0], self.shape[0][1], self.shape[0][2], self.shape[0][3]]
    info = tensor
    tensor = self.YUV_linear_initializtion(tensor, _shape)   
    tensor = tensor[:, :self.shape[0][1],  :self.shape[0][2], :]  
    init_output = x3 = tensor
    x1 = self.deeprec_list[0](tensor)
    deep_output = []
    for i in range(int(self.depth)):
      x2 = self.deeprec_list[i*4+1](x1)
      x2 = self.deeprec_list[i*4+2](x2)
      x1 = x1 + x2
      x3 = self.deeprec_list[i*4+3](x1) + x3
      analysis_t = self.YUV_Measurement(x3)
      diff = analysis_t - info
      diff = self.YUV_measurement_transpose(diff, _shape)
      x3 = x3 - self.rho[i]*diff
      if i<(self.depth-1):
        x1 = self.deeprec_list[i*4+4](x3) + x1
      deep_output.append(x3)
      
    return init_output, deep_output        
           
  def build_shape(self, _shape):
    self.shape = []
    if _shape.shape==4:
      self.shape.append(_shape)
    else:
      self.shape.append([1, _shape[0], _shape[1], 3])   
      
  def call(self, tensor, reverse=False):
    if not reverse:
      self.build_shape(tf.shape(tensor))
      return self.normflow(tensor)
    else:
      return self.reverseflow(tensor)  
    
class HyperAnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            HY_NUM, (1, 1), name="layer_0", corr=True, strides_down=1,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(HyperAnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    #pad
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor

class Add_Mean(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, idx, *args, **kwargs):
    self.idx = idx
    super(Add_Mean, self).__init__(*args, **kwargs)

  def build(self, input_shape):

    self.mean = self.add_variable(
        "mean", shape= (1, 1, 1, _NUM - BASE_NUM), dtype=self.dtype,
        initializer=tf.truncated_normal_initializer(stddev=0.2))
    super(Add_Mean, self).build(input_shape)

  def call(self, tensor):
    #pad
    zero = tf.zeros_like(tensor)
    mean = zero[:, :, :, :_NUM-BASE_NUM-self.idx[0]] + self.mean[:, :, :, self.idx[0]:]
    tensor = tf.concat([tensor, mean], axis=3)
    return tensor

class HyperSynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            _NUM, (1, 1), name="layer_0", corr=False, 
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=None),
    ]
    super(HyperSynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor

class Quantize(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    self._default_quantize_param = parameterizers.NonnegativeParameterizer()
    super(Quantize, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.quantize = self._default_quantize_param(
          name="gamma", shape=[1, 1, 1, self.num_filters], dtype=self.dtype,
          getter=self.add_variable,
          initializer=tf.constant_initializer(1.0)) 
    super(Quantize, self).build(input_shape)

  def call(self, tensor, quantized = True):
    if quantized:
      tensor = tensor / self.quantize
    else:
      tensor = tensor * self.quantize
    return tensor

class RGB2YUV(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, *args, **kwargs):
    super(RGB2YUV, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    z1 = 0.5092
    z2 = 0.19516
    WR = z1/(1+z1+z2)
    WG = 1/(1+z1+z2)
    WB = z2/(1+z1+z2)
    C = np.array([[WR, WG, WB], [-WR/(1-WB)/2, -WG/(1-WB)/2, 0.50000], [0.50000, -WG/(1-WR)/2, -WB/(1-WR)/2]], np.float32)
    self.trans_m = tf.convert_to_tensor(C, dtype=self.dtype)
    self.inv_trans_m = tf.convert_to_tensor(np.linalg.inv(C), dtype=self.dtype)
    super(RGB2YUV, self).build(RGB2YUV)

  def call(self, tensor, reverse=False):
    if not reverse:
      tensor = tf.matmul(tensor, tf.transpose(self.trans_m))
    else:
      tensor = tf.matmul(tensor, tf.transpose(self.inv_trans_m))  
    return tensor 

class Learn_RGB2YUV(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, *args, **kwargs):
    self.create_weight = parameterizers.NonnegativeParameterizer(
    minimum=1e-2)
    super(Learn_RGB2YUV, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.weight = self.create_weight(
          name="weight", shape=[2], dtype=self.dtype,
          getter=self.add_variable, initializer=tf.initializers.ones()) 
    self.one = tf.ones((1)) 
    self.RGB_weight = tf.concat([self.weight, self.one], 0)   
    line1 = self.RGB_weight/tf.reduce_sum(self.RGB_weight) 
    e2 = tf.constant([0, 0, 1], dtype = self.dtype) 
    line2 = (e2-line1)/(1-line1[2])/2 
    e0 = tf.constant([1, 0, 0], dtype = self.dtype) 
    line3 = (e0-line1)/(1-line1[0])/2 
    self.trans_m = tf.stack([line1, line2, line3], axis=0) 
    self.inv_trans_m = tf.linalg.inv(self.trans_m) 

    super(Learn_RGB2YUV, self).build(Learn_RGB2YUV)    

  def call(self, tensor, reverse=False):
    if not reverse:
      tensor = tf.matmul(tensor, tf.transpose(self.trans_m))
    else:
      tensor = tf.matmul(tensor, tf.transpose(self.inv_trans_m))  
    return tensor   
    
def train(args):
  """Trains the model."""
  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)
  graph = tf.Graph()
  # Create input data pipeline.
  with tf.device("/cpu:0"):
    dataset = '../Cityscapes_dataset' # Select your path

    IMG_TRAIN_LIST = os.path.join(dataset, 'new_img_train.txt')
    IMG_VAL_LIST = os.path.join(dataset, 'new_img_val.txt')
    IMG_TEST_LIST = os.path.join(dataset, 'new_img_val.txt')

    ANNO_TRAIN_LIST = os.path.join(dataset, 'new_anno_train.txt')
    ANNO_VAL_LIST = os.path.join(dataset, 'new_anno_val.txt') 
    ANNO_TEST_LIST = os.path.join(dataset, 'new_anno_val.txt')
    f = open(IMG_TRAIN_LIST)
    lines = f.readlines()
    train_files = [line.strip() for line in lines]
    train_files = glob.glob(os.path.join(dataset, "crop_train/*.png"))
    pdb.set_trace()
    #train_files = glob.glob(args.train_glob)
    #label_files = [filename.replace(CITYSCAPE_IMG_DIR, CITYSCAPE_ANNO_DIR) for filename in train_files]
    #label_files = [filename.replace('_leftImg8bit.png', '_gtFine_labelIds2.png') for filename in label_files]
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()
  #pdb.set_trace()
  #x = tf.image.resize_images(img, (50, 50), tf.image.ResizeMethod.BILINEAR, False)
  #x = tf.image.resize_images(x, (96, 96), tf.image.ResizeMethod.BILINEAR, False)
  #labels = x[:, :, :, -1:]
  #labels = tf.cast(labels, tf.int32)
  #x = x[:, :, :, :3]

  # Instantiate model.
  rgb2yuv = Learn_RGB2YUV()
  analysissynthesis_transform = AnalysisSynthesisTransform(args.num_filters, depth=2)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
  #add_mean = Add_Mean(idx)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  quantizefunc = Quantize(_NUM)

  ori_x = x
  x_shape = tf.shape(x)
  x = Padding(x, analysissynthesis_transform.block_width)
  yuv_x = rgb2yuv(x, False)
  # Build autoencoder and hyperprior.
  y = analysissynthesis_transform(yuv_x, False)
  y_tilde, y_likelihoods = entropy_bottleneck(quantizefunc(y), training=True)
  init_output, deep_output = analysissynthesis_transform(quantizefunc(y_tilde, False), True)
  # Total number of bits divided by number of pixels.
  train_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) ) / (-np.log(2) * num_pixels) 
               
  # Mean squared error across pixels.
  train_mse = 0
  train_mse += tf.reduce_mean(tf.squared_difference(ori_x, rgb2yuv(init_output, True)[:, :x_shape[1], :x_shape[2], :]))
  for i in range(2):
    if i==1:
      train_last_mse = tf.reduce_mean(tf.squared_difference(ori_x, rgb2yuv(deep_output[i], True)[:, :x_shape[1], :x_shape[2], :]))
    train_mse += tf.reduce_mean(tf.squared_difference(ori_x, rgb2yuv(deep_output[i], True)[:, :x_shape[1], :x_shape[2], :]))
  train_mse /= 3
  # Multiply by 255^2 to correct for rescaling.
  
  train_mse *= 255 ** 2
  train_last_mse *= 255**2
  
  # The rate-distortion cost.
  train_loss = args.lmbda * train_mse + args.eta * train_bpp 
  #g = tf.gradients(train_loss, [img])
  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
  variables = tf.contrib.framework.get_variables_to_restore()
  variables_to_restore = [v for v in variables if v.name.split('/')[0] not in ('global_step:0')]
  #variables_to_restore = [v for v in variables if v.name.split('/')[0] not in ('hyper_synthesis_transform', 'analysis_synthesis_transform', 'hyper_analysis_transform', 'entropy_bottleneck', 'gaussian_conditional', 'analysis_synthesis_transform_1', 'global_step:0', 'quantize', 'add__mean')]
  #variables_to_restore = [v for v in variables if v.name.split('/')[0] in ('analysis_synthesis_transform_1') and v.name.split('/')[1][:6]=='layer3']
  #variables_to_restore = [v for v in variables if v.name.split('/')[0] not in ('global_step:0', 'learn_rg_b2yuv')]
  #var_train = [v for v in variables if v.name.split('/')[0] in ('analysis_synthesis_transform_1')]
  #var_train2 = [v for v in var_train if v.name.split('/')[1] in ('rho:0')]
  #tf.summary.scalar("loss", train_loss)
  #tf.summary.scalar("bpp", train_bpp)
  #tf.summary.scalar("mse", train_mse)

  #tf.summary.image("original", quantize_image(x))
  #tf.summary.image("reconstruction", quantize_image(deep_output[4]))
  ckpt1 = tf.train.get_checkpoint_state('model/CS_IoT_19_kitti_16_depth_2_without_mean_33_yuv_v4_learn_yuv_w10_without_hype_reshape_lmb0007_eta13_48_28_10')
  saver1 = tf.train.Saver(variables_to_restore)
  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]
  avg_bpp = AverageMeter(args.record_length)
  avg_mse = AverageMeter(args.record_length)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  count = 0
  with tf.train.MonitoredTrainingSession(
    hooks=hooks, checkpoint_dir=args.checkpoint_dir,
    save_checkpoint_secs=100, config=config) as sess:
    i = 0  
    if args.init:
      saver1.restore(sess, ckpt1.model_checkpoint_path)
    pdb.set_trace()
    print(sess.run([rgb2yuv.weight]))  
    while not sess.should_stop():
      bpp, mse, last_mse, _, _, _ = sess.run([train_bpp, train_mse, train_last_mse, main_step, aux_step, entropy_bottleneck.updates[0]])
      avg_bpp.update(bpp)
      avg_mse.update(last_mse)
      if i%args.info_step==0:          
        logging.info('step %d avg bpp %.4f avg mse %.4f'%(i, avg_bpp.avg, avg_mse.avg))
      i += 1

def compress2(args):
  """Compresses an image."""
  with tf.device("/cpu:0"):
    dataset = '../Cityscapes_dataset' # Select your path

    IMG_TRAIN_LIST = os.path.join(dataset, 'new_img_train.txt')
    IMG_VAL_LIST = os.path.join(dataset, 'new_img_val.txt')
    IMG_TEST_LIST = os.path.join(dataset, 'new_img_val.txt')

    ANNO_TRAIN_LIST = os.path.join(dataset, 'new_anno_train.txt')
    ANNO_VAL_LIST = os.path.join(dataset, 'new_anno_val.txt') 
    ANNO_TEST_LIST = os.path.join(dataset, 'new_anno_val.txt')
    f = open(IMG_VAL_LIST)
    lines = f.readlines()
    train_files = [line.strip() for line in lines]
    num = len(train_files)
    #train_files = glob.glob(args.input_file)
    #label_files = [filename.replace(CITYSCAPE_IMG_DIR, CITYSCAPE_ANNO_DIR) for filename in train_files]
    #label_files = [filename.replace('_leftImg8bit.png', '_gtFine_labelIds2.png') for filename in label_files]
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.input_file))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=4)
    train_dataset = train_dataset.batch(1)
    train_dataset = train_dataset.prefetch(32)
  # Load input image and add batch dimension.
  x = train_dataset.make_one_shot_iterator().get_next()
  x.set_shape([1, None, None, 3])
  x_shape = tf.shape(x)
  x = x[:, :x_shape[1]//16*16, :x_shape[2]//16*16, :]
  #labels = x[:, :, :, -1:]
  #labels = tf.cast(labels, tf.int32)
  #labels.set_shape([1, None, None, 1])
  #x = x[:, :, :, :-1]
  x.set_shape([1, None, None, 3])
  

  # Instantiate model.
  rgb2yuv = Learn_RGB2YUV()
  analysissynthesis_transform = AnalysisSynthesisTransform(args.num_filters, depth=2)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
  #add_mean = Add_Mean(idx)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  quantizefunc = Quantize(_NUM)

  ori_x = x
  x = Padding(x, analysissynthesis_transform.block_width)
  yuv_x = rgb2yuv(x)
  # Transform and compress the image.
  y = analysissynthesis_transform(yuv_x, False)
  y_shape = tf.shape(y)
  string = entropy_bottleneck.compress(quantizefunc(y))

  # Transform the quantized image back (if requested).
  y_hat, y_likelihoods = entropy_bottleneck(quantizefunc(y), training=False)
  _, deep_output = analysissynthesis_transform(quantizefunc(y_hat, False), True)

  x_hat = rgb2yuv(deep_output[-1], True)
  x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = (tf.reduce_sum(tf.log(y_likelihoods))) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  ori_x *= 255
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)

  mse = tf.reduce_mean(tf.squared_difference(ori_x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x_hat, ori_x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, ori_x, 255))
  # [Variable and model creation goes here.]
  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    
    tensors = [string, 
               tf.shape(x)[1:-1], tf.shape(y)[1:-1]]
    
            
    # Write a binary file with the shape information and the compressed string.

    # If requested, transform the quantized image back and measure performance.
    sum_psnr = 0
    sum_msssim = 0
    sum_eval_bpp = 0
    sum_bpp = 0
    #num=749
    for i in range(num):
      _eval_bpp, _mse, _psnr, _msssim, _num_pixels, arrays = sess.run(
          [eval_bpp, mse, psnr, msssim, num_pixels, tensors])
      packed = tfc.PackedTensors()    
      packed.pack(tensors, arrays)
      # The actual bits per pixel including overhead.
      bpp = len(packed.string) * 8 / _num_pixels
      sum_psnr += _psnr
      sum_msssim += _msssim
      sum_eval_bpp += _eval_bpp
      sum_bpp += bpp
      print('sdjdksjdk'+str(i))
      if i==num-1:
        print("PSNR (dB): {:0.2f}".format(sum_psnr/num))
        print("Multiscale SSIM: {:0.4f}".format(sum_msssim/num))
        print("Information content in bpp: {:0.4f}".format(sum_eval_bpp/num))
        print("Actual bits per pixel: {:0.4f}".format(sum_bpp/num))
      if i%100==0:
        pass  

def decompress(args):
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  string = tf.placeholder(tf.string, [1])
  side_string = tf.placeholder(tf.string, [1])
  x_shape = tf.placeholder(tf.int32, [2])
  y_shape = tf.placeholder(tf.int32, [2])
  z_shape = tf.placeholder(tf.int32, [2])
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = [string, side_string, x_shape, y_shape, z_shape]
  arrays = packed.unpack(tensors)

  # Instantiate model.
  synthesis_transform = SynthesisTransform(args.num_filters)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)

  # Decompress and transform the image back.
  z_shape = tf.concat([z_shape, [args.num_filters]], axis=0)
  z_hat = entropy_bottleneck.decompress(
      side_string, z_shape, channels=args.num_filters)
  sigma = hyper_synthesis_transform(z_hat)
  sigma = sigma[:, :y_shape[0], :y_shape[1], :]
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(
      sigma, scale_table, dtype=tf.float32)
  y_hat = conditional_bottleneck.decompress(string)
  x_hat = synthesis_transform(y_hat)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = write_png(args.output_file, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op, feed_dict=dict(zip(tensors, arrays)))


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true", default=True, 
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--group", action="store_true", default=True, 
      help="Group compress.") 
  parser.add_argument(
      "--init", action="store_true", default=False, 
      help="initialize or not.")          
  parser.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters per layer.")
  parser.add_argument(
      "--record_length", type=int, default=1000,
      help="Number of value to record.")    
  parser.add_argument(
      "--info_step", type=int, default=50,
      help="Number of steps to info.")    
  parser.add_argument(
      "--output_stride", type=int, default=8,
      help="output stride in the resnet model.")   
  parser.add_argument("--classes", default=19, help="The ignore label value.")      
  parser.add_argument(
      "--structure", type=dict, default={'0': 1, '1':1, '2':2},
      help="structure of analysis network.")   
  parser.add_argument('--rgb_mean', default=[72.39239876,82.90891754,73.15835921], help='RGB mean value of ImageNet.')
  parser.add_argument('--scales', default=[0.5,0.75,1.0,1.25,1.5,1.75,2.0], help='Scales for random scale.')       
  parser.add_argument("--pretrained_resnet", default="resnet_v2_101_2017_04_14/resnet_v2_101.ckpt", help="Path to save pretrained model.")     
  parser.add_argument("--pretrained_seg", default="model/seg", help="Path to restore segmantation checkpoint.")
  parser.add_argument("--pretrained_CS", default="model/CS_IoT_6_width8_r_05", help="Path to restore CS checkpoint.")  
  parser.add_argument("--pretrained_IoT", default="model/CS_IoT_7_eta_06_gamma_2", help="Path to restore CS checkpoint.")    
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=30,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=96,
      help="Size of image patches for training.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.05, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--gamma", type=float, default=0.1, dest="gamma",
      help="gamma for segmentation loss.")    
  train_cmd.add_argument(
      "--eta", type=float, default=1.0, dest="eta",
      help="eta for rate-distortion tradeoff.")       
  train_cmd.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "--input_file",
        help="Input filename.")
    cmd.add_argument(
        "--output_file", nargs="?",
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))
  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    if args.group:
      compress2(args)
    else:     
      compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
