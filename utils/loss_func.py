import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
import sys
import cv2
import os

content_layers = ['h_conv4_2']
style_layers = ['h_relu1_1', 'h_relu2_1', 'h_relu3_1', 'h_relu4_1', 'h_relu5_1']
style_weight = [.2, .2, .2, .2, .2]

alpha = 5e0
beta = 1e2
theta = 1e-2

iter_num = 1000
learning_rate = 1e0

def content_loss(gen, base, mode=0):
    """
        Define the content loss for the image
    """
    _, h, w, f = gen.get_shape()
    M = h.value * w.value
    N = f.value
    if mode == 0:
        params = 1. / (2. * N**0.5 * M**0.5)
    if mode == 1:
        _, h, w, f = gen.get_shape()
        params = 1. / (h * w)
    if mode == 2:
        params = 1. / 2.
    loss = params * tf.reduce_sum(tf.pow((gen - base), 2))
    return loss

def style_loss(gen, base, mode=0):
    """
        Define the style loss for the image
    """
    _, h, w, f = gen.get_shape()
    M = h.value * w.value
    N = f.value
    if mode == 0:
        gram_gen = gram_matrix(gen)
        gram_base = gram_matrix(base)
        loss_scale = 1. / (4 * M**2 * N**2)
        loss = loss_scale * tf.reduce_sum(tf.pow((gram_gen - gram_base), 2))
    if mode == 1:
        avg_gen = avg_flat(gen)
        avg_base = avg_flat(base)
        loss_scale = 1. / (4 * M**2 * N**2)
        loss = loss_scale * tf.reduce_sum(tf.pow((avg_gen - avg_base), 2))
    return loss

def de_noise(gen):
    loss = tf.reduce_sum(tf.abs(gen[:,1:,:,:] - gen[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(gen[:,:,1:,:] - gen[:,:,:-1,:]))
    return loss

def wrap_content_loss(sess, vggnet, content_img):
    """
        Accumulate all content loss from suggested layers
    """
    base = sess.run(vggnet['input'].assign(content_img))
    loss = 0.
    for lay in content_layers:
        base_layer = sess.run(vggnet[lay])
        base_layer = tf.convert_to_tensor(base_layer)
        # print(base_layer.eval())
        gen_layer = vggnet[lay]
        # print(gen_layer.eval())
        loss += content_loss(gen_layer, base_layer)
    loss /= float(len(content_layers))
    return loss

def wrap_style_loss(sess, vggnet, style_img):
    """
        Accumulate all style loss from suggested layers
    """
    base = sess.run(vggnet['input'].assign(style_img))
    loss = 0.
    for lay, wei in zip(style_layers, style_weight):
        base_layer = sess.run(vggnet[lay])
        base_layer = tf.convert_to_tensor(base_layer)
        gen_layer = vggnet[lay]
        loss += style_loss(gen_layer, base_layer) * wei
    loss /= float(len(style_layers))
    return loss
    
def gram_matrix(tensor):
    num_f = tensor.get_shape().as_list()[-1]
    tensor_flat = tf.reshape(tensor, [-1, num_f])
    gram_out = tf.matmul(tf.transpose(tensor_flat), tensor_flat)
    return gram_out

def avg_flat(tensor):
    # num_f = tensor.get_shape().as_list()[-1]
    tensor_flat = tf.reshape(tensor, [-1])
    return tensor_flat