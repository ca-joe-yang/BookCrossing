#! python3

import os
import time

import numpy as np
import tensorflow as tf

from .BasicModel import BasicModel

slim = tf.contrib.slim

class Baseline(BasicModel):

    def build(self, mu, N, M, verbose=True):
        if verbose:
            start_time = time.time()
            print('[*] Building model...')

        self.mu = mu
        self.N = N
        self.M = M

        #self.is_training = tf.placeholder(tf.bool, name='is_training')
        #self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        #self.weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        self.bu = tf.get_variable(name='bu', shape=[N], 
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))
        self.bi = tf.get_variable(name='bi', shape=[M], 
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32))

        # Inputs: [R, G, R] scaled [-1, 1]
        self.user_ids = tf.placeholder(tf.int32, shape=[None], name='user_ids')
        self.item_ids = tf.placeholder(tf.int32, shape=[None], name='item_ids')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        self.preds = self.mu + tf.nn.embedding_lookup(self.bu, self.user_ids) + tf.nn.embedding_lookup(self.bi, self.item_ids)
        
        if verbose:
            print(('[*] Model built: %ds' % (time.time() - start_time)))

    def build_loss(self):
        with tf.variable_scope('loss'):
            mae_loss = tf.reduce_mean(tf.abs( tf.cast(self.labels, tf.float32) - self.preds ))
            #reg_loss = tf.reduce_sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) )
            self.loss = mae_loss
        tf.summary.scalar('Loss', self.loss)

        with tf.variable_scope('accuracy'):
            corrects = tf.equal( tf.cast(self.preds, tf.int32), self.labels )
            self.acc = tf.reduce_mean( tf.cast(corrects, dtype=tf.float32) )
        tf.summary.scalar('Accuracy', self.acc)

    def build_optimizer(self, optimizer, lr):
        print('[*] lr={}'.format(lr))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            self.optim = self.optimizer.minimize(self.loss)

    def train_valid(self, user_ids, item_ids, labels):        
        if is_training:
            writer = self.train_writer
        else:
            writer = self.valid_writer

        feed_dict = {
            self.user_ids: user_ids,
            self.book_ids: item_ids,
            self.labels: labels,
        }
            
        if is_training:
            summary, b_loss, b_acc, _ = self.sess.run(
                [self.summary, self.loss, self.acc, self.optim], 
                feed_dict=feed_dict,
            )
        else:
            summary, b_loss, b_acc = self.sess.run(
                [self.summary, self.loss, self.acc], 
                feed_dict=feed_dict,
            )
        if self.step_idx % 10 == 0:
            writer.add_summary(summary, self.step_idx)
        self.step_idx += 1
        
        return b_loss, b_acc

    def predict(self, user_ids, item_ids):

        feed_dict = {
            self.user_ids: user_ids,
            self.book_ids: item_ids,
        }
            
        b_preds = self.sess.run(
            self.preds, 
            feed_dict=feed_dict,
        )
        
        return b_preds
