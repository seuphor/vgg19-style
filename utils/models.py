import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
import sys
import cv2
import os

vgg_weights = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
# print(vgg_weights.keys())
vgg_layers = vgg_weights['layers'][0]

content_layers = ['h_conv4_2']
style_layers = ['h_relu1_1', 'h_relu2_1', 'h_relu3_1', 'h_relu4_1', 'h_relu5_1']
style_weight = [.2, .2, .2, .2, .2]

alpha = 5e0
beta = 1e2
theta = 1e-2

iter_num = 1000
learning_rate = 1e0

def load_model(input_):
    """
        Reconstruct the VGG19 models using weights from pre-trained model
    """
    _, height, width, channel = input_.shape
    vgg_style = {}
    
    vgg_style['input'] = tf.Variable(np.zeros((1, height, width, channel)), dtype=tf.float32)
    
    # restore layer 1
    vgg_style['h_conv1_1'] = conv2d(vgg_style['input'], w=load_weights(vgg_layers, 0), name='h_conv1_1')
    vgg_style['h_relu1_1'] = relu(vgg_style['h_conv1_1'], b=load_biases(vgg_layers, 0), name='h_relu1_1')
    
    vgg_style['h_conv1_2'] = conv2d(vgg_style['h_relu1_1'], w=load_weights(vgg_layers, 2), name='h_conv1_2')
    vgg_style['h_relu1_2'] = relu(vgg_style['h_conv1_2'], b=load_biases(vgg_layers, 2), name='h_relu1_2')
    
    vgg_style['h_pool1'] = pool(vgg_style['h_relu1_2'], name='h_pool1')
    
    # restore layer 2
    vgg_style['h_conv2_1'] = conv2d(vgg_style['h_pool1'], w=load_weights(vgg_layers, 5), name='h_conv2_1')
    vgg_style['h_relu2_1'] = relu(vgg_style['h_conv2_1'], b=load_biases(vgg_layers, 5), name='h_relu2_1')
    
    vgg_style['h_conv2_2'] = conv2d(vgg_style['h_relu2_1'], w=load_weights(vgg_layers, 7), name='h_conv2_2')
    vgg_style['h_relu2_2'] = relu(vgg_style['h_conv2_2'], b=load_biases(vgg_layers, 7), name='h_relu2_2')
    
    vgg_style['h_pool2'] = pool(vgg_style['h_relu2_2'], name='h_pool2')
    
    # restore layer 3
    vgg_style['h_conv3_1'] = conv2d(vgg_style['h_pool2'], w=load_weights(vgg_layers, 10), name='h_conv3_1')
    vgg_style['h_relu3_1'] = relu(vgg_style['h_conv3_1'], b=load_biases(vgg_layers, 10), name='h_relu3_1')
    
    vgg_style['h_conv3_2'] = conv2d(vgg_style['h_relu3_1'], w=load_weights(vgg_layers, 12), name='h_conv3_2')
    vgg_style['h_relu3_2'] = relu(vgg_style['h_conv3_2'], b=load_biases(vgg_layers, 12), name='h_relu3_2')
    
    vgg_style['h_conv3_3'] = conv2d(vgg_style['h_relu3_2'], w=load_weights(vgg_layers, 14), name='h_conv3_3')
    vgg_style['h_relu3_3'] = relu(vgg_style['h_conv3_3'], b=load_biases(vgg_layers, 14), name='h_relu3_3')
    
    vgg_style['h_conv3_4'] = conv2d(vgg_style['h_relu3_3'], w=load_weights(vgg_layers, 16), name='h_conv3_4')
    vgg_style['h_relu3_4'] = relu(vgg_style['h_conv3_4'], b=load_biases(vgg_layers, 16), name='h_relu3_4')
    
    vgg_style['h_pool3'] = pool(vgg_style['h_relu3_2'], name='h_pool3')
    
    # restore layer 4
    vgg_style['h_conv4_1'] = conv2d(vgg_style['h_pool3'], w=load_weights(vgg_layers, 19), name='h_conv4_1')
    vgg_style['h_relu4_1'] = relu(vgg_style['h_conv4_1'], b=load_biases(vgg_layers, 19), name='h_relu4_1')
    
    vgg_style['h_conv4_2'] = conv2d(vgg_style['h_relu4_1'], w=load_weights(vgg_layers, 21), name='h_conv4_2')
    vgg_style['h_relu4_2'] = relu(vgg_style['h_conv4_2'], b=load_biases(vgg_layers, 21), name='h_relu4_2')
    
    vgg_style['h_conv4_3'] = conv2d(vgg_style['h_relu4_2'], w=load_weights(vgg_layers, 23), name='h_conv4_3')
    vgg_style['h_relu4_3'] = relu(vgg_style['h_conv4_3'], b=load_biases(vgg_layers, 23), name='h_relu4_3')
    
    vgg_style['h_conv4_4'] = conv2d(vgg_style['h_relu4_3'], w=load_weights(vgg_layers, 25), name='h_conv4_4')
    vgg_style['h_relu4_4'] = relu(vgg_style['h_conv4_4'], b=load_biases(vgg_layers, 25), name='h_relu4_4')
    
    vgg_style['h_pool4'] = pool(vgg_style['h_relu4_4'], name='h_pool4')
    
    # restore layer 5
    vgg_style['h_conv5_1'] = conv2d(vgg_style['h_pool4'], w=load_weights(vgg_layers, 28), name='h_conv5_1')
    vgg_style['h_relu5_1'] = relu(vgg_style['h_conv5_1'], b=load_biases(vgg_layers, 28), name='h_relu5_1')
    
    vgg_style['h_conv5_2'] = conv2d(vgg_style['h_relu5_1'], w=load_weights(vgg_layers, 30), name='h_conv5_2')
    vgg_style['h_relu5_2'] = relu(vgg_style['h_conv5_2'], b=load_biases(vgg_layers, 30), name='h_relu5_2')
    
    vgg_style['h_conv5_3'] = conv2d(vgg_style['h_relu5_2'], w=load_weights(vgg_layers, 32), name='h_conv5_3')
    vgg_style['h_relu5_3'] = relu(vgg_style['h_conv5_3'], b=load_biases(vgg_layers, 32), name='h_relu5_3')
    
    vgg_style['h_conv5_4'] = conv2d(vgg_style['h_relu5_3'], w=load_weights(vgg_layers, 34), name='h_conv5_4')
    vgg_style['h_relu5_4'] = relu(vgg_style['h_conv5_4'], b=load_biases(vgg_layers, 34), name='h_relu5_4')
    
    vgg_style['h_pool5'] = pool(vgg_style['h_relu5_4'], name='h_pool5')    
        
    return vgg_style

def conv2d(input_, w, name):
    h_conv = tf.nn.conv2d(input_, w, strides=[1,1,1,1], padding='SAME', name=name)    
    return h_conv
    
def relu(input_, b, name):
    b = tf.reshape(b, [-1,])
    # print(b.get_shape())
    h_relu = tf.nn.relu(input_ + b, name=name)
    return h_relu

def pool(input_, name, pool='avg'):
    if pool == 'avg':
        h_pool = tf.nn.avg_pool(input_, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
    if pool == 'max':
        h_pool = tf.nn.max_pool(input_, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
    return h_pool

def load_weights(layers, idx):
    """
        Loads the layer weight by layer index
    """
    w_ = layers[idx][0][0][2][0][0]
    w = tf.constant(w_)
    return w

def load_biases(layers, idx):
    """
        Loads the layer bias by layer index
    """
    b_ = layers[idx][0][0][2][0][1]
    b = tf.constant(b_)
    return b
