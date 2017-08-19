from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


FLAGS = None

def deepnn(x):
    """
    deepn builds the graph for deep net for classifying digits.

    Args:
    x: an input teensors with the dimension (N_examples, 784), where 784 is the
    number of pixels in the a standard MNIST image.

    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_exmaples, 10), with values
    equal to the logitis of the classifingthe digit in one of 10 classes (the
    digits being 0-9), keep_prob is a scalar placeholder for the probability of
    dropout.
    """

    # Reshape to use within a convnet.
    # LAst dimension is for "features" -  there is only one here,
    #since images are grayscale -- it would be 3 for and RGB imags,
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x,[-1, 28, 28, 1])

    # First convonet layer- maps pone grayscale images to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variables([5, 5,1,32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2(x_image, W_conv1)+b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolution layer - maps 32 features maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 rounds of downsaampling, our 28x28
    # image is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.mathmul(h_pool2_flat, W_fc1)+b_fc1)

    
