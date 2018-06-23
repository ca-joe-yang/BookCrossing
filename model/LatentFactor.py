#! python3

import os
import time

import numpy as np
import tensorflow as tf

from .Baseline import Baseline
from .model_utils import *

slim = tf.contrib.slim

class LatentFactor(Baseline):

    def build(self, mu, N, M, latent_dim=20, verbose=True):
        if verbose:
            start_time = time.time()
            print('[*] Building model...')

        self.mu = mu
        self.N = N
        self.M = M

        #self.is_training = tf.placeholder(tf.bool, name='is_training')
        #self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        self.bu = tf.get_variable(name='bu', shape=[N], 
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32),
            regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))
        
        self.bi = tf.get_variable(name='bi', shape=[M], 
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32),
            regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))

        # user-factors vector
        self.p = tf.get_variable(name='p', shape=[N, latent_dim], 
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32),
            regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))

        # item-factors vector
        self.q = tf.get_variable(name='q', shape=[M, latent_dim], 
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32),
            regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))


        # Inputs: [R, G, R] scaled [-1, 1]
        self.user_ids = tf.placeholder(tf.int32, shape=[None], name='user_ids')
        self.item_ids = tf.placeholder(tf.int32, shape=[None], name='item_ids')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        user_bias = tf.nn.embedding_lookup(self.bu, self.user_ids)
        item_bias = tf.nn.embedding_lookup(self.bi, self.item_ids)

        user_latent = tf.nn.embedding_lookup(self.p, self.user_ids)
        item_latent = tf.nn.embedding_lookup(self.q, self.item_ids)

        self.preds = self.mu + user_bias + item_bias + tf.einsum('ij,ij->i', user_latent, item_latent)
        
        if verbose:
            print(('[*] Model built: %ds' % (time.time() - start_time)))

