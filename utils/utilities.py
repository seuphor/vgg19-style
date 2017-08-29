import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.io
import sys
import cv2
import os

avg_values = np.array([123.68, 116.779, 103.939]).reshape(1,1,1,3)

def read_img(path):
    # read initial image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    return img

def resize_style(content, style):
    # resize the style img to fit content img
    if content.shape[0] == style.shape[0] and content.shape[1] == style.shape[1]:
        print('same size...')
        return style
    else:
        resize_style = cv2.resize(style, (content.shape[1], content.shape[0]), interpolation=cv2.INTER_AREA)
        print('Resized... content image size: {}, {} / style image size: {}, {}'.format(content.shape[0], content.shape[1],
                                                                                        resize_style.shape[0], resize_style.shape[1]))
        del style
        return resize_style
    
def pre_process(img):
    img = np.float32(img)
    # expand one more dim and substract avg_value to get zero-center
    img = np.expand_dims(img, axis=0)
    img -= avg_values
    return img

def post_process(img):
    img += avg_values
    img = img[0]
    
    # clip to (0, 255) switch the dtype to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def cvt2original_color(target, x, mode='luv'):
    """
        Use Opencv default function to map original color of target image to x
    """    
    # resize style img
    x = resize_style(target, x)
    
    if mode == 'luv':
        cvt = cv2.COLOR_RGB2LUV
        inv = cv2.COLOR_LUV2RGB
    
    target_ = cv2.cvtColor(target, cvt)
    x_ = cv2.cvtColor(x, cvt)
    
    f1_tar, f2, f3 = cv2.split(target_)
    f1_x, _, _ = cv2.split(x_)
    
    # f1_x = (np.std(f1_tar) / np.std(f1_x)) * (f1_x - np.mean(f1_x)) + np.mean(f1_tar)
    # f1_x = f1_x.astype(np.uint8)
    new = cv2.merge((f1_x, f2, f3))
    new_rgb = cv2.cvtColor(new, inv)
    # new_rgb = new_rgb.astype(np.uint8)
    return new_rgb