#! python3

import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def inputs_preprocssing(X, mode):
    # X \in [0, 255], RGB
    if mode == 'inception':
        return normalize_range(X, [0, 255], [-1, 1])
    elif mode == 'vgg':
        vgg_bgr_mean = [103.939, 116.779, 123.68]
        X_r, X_g, X_b = np.split(X, indices_or_sections=3, axis=3)
        X_b -= vgg_bgr_mean[0]
        X_g -= vgg_bgr_mean[1]
        X_r -= vgg_bgr_mean[2]
        return np.concat([X_b, X_g, X_r], axis=3)

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)] 

def shuffle_n(n):
    x = list(range(n))
    random.shuffle(x)

    return x
def shuffle_xy(x, y):
    assert(len(x) == len(y))
    a = x.copy()
    b = y.copy()  
    
    combined = list(zip(a, b))
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)
    return a, b

def normalize_range(x, range_1, range_2):
    x = x.astype(float)
    a,b = range_1
    c,d = range_2
    return (x-a)*(d-c)/(b-a) + c

def bilinear_initializer(shape):
    assert(shape[0] == shape[1])
    assert(shape[2] == shape[3])
    
    k_size = shape[0]
    f = np.ceil(k_size/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    
    bilinear = np.zeros([k_size, k_size])
    for i in range(k_size):
        for j in range(k_size):
            bilinear[i, j] = (1 - abs(i / f - c)) * (1 - abs(j / f - c))

    bilinear_weights = np.zeros(shape)
    for i in range(shape[2]):
        bilinear_weights[:, :, i, i] = bilinear

    return tf.constant_initializer(value=bilinear_weights, dtype=tf.float32)

def weighted_sparse_softmax_cross_entropy_with_logits(logits, labels, weights, n_class):
    with tf.name_scope('loss'):
        epsilon = tf.constant(value=1e-10)
        
        logits_flat = tf.reshape(logits, (-1, n_class))
        logits_flat += epsilon
        logits_softmax = tf.nn.softmax(logits_flat)

        labels_flat = tf.reshape(labels, (-1, 1))
        target = tf.one_hot(labels_flat, depth=n_class)
        target_flat = tf.reshape(target, (-1, n_class))

        cross_entropy = -tf.multiply(target_flat * tf.log(logits_softmax + epsilon), weights)
    return cross_entropy

def get_var_list_in_checkpoint_file(filename, all_tensors=True, tensor_name=None):
    var_list=[]
    reader = pywrap_tensorflow.NewCheckpointReader(filename)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            var_list.append(key)
    else:
        var_list.append(tensor_name)
    return var_list
