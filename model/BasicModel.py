#! python3

import tensorflow as tf
import numpy as np
import sys, os, time
from .model_utils import *

slim = tf.contrib.slim

class BasicModel():

    def __init__(self, sess, model_name, checkpoint_dirname):
        self.sess = sess
        self.model_name = model_name
        self.checkpoint_dirname = checkpoint_dirname

        self.step_idx = 0

    def model_class(self):
        return self.__class__.__name__

    def model_checkpoint_dirname(self):
        return os.path.join(self.checkpoint_dirname, self.model_class())

    def initialize(self):
        print('[*] Initializing Variables')
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _print_shape(self, x):
        x = tf.Print(x, [tf.shape(x)], message="This is a: ")
        return x

    def save(self):
        if not os.path.exists(self.checkpoint_dirname):
            os.makedirs(self.checkpoint_dirname)

        if not os.path.exists(self.model_checkpoint_dirname()):
            os.makedirs(self.model_checkpoint_dirname())
         
        model_checkpoint_filename = os.path.join(self.model_checkpoint_dirname(), self.model_name)
        print('[*] Saving checkpoints of to {}'.format(model_checkpoint_filename))
        
        saver = tf.train.Saver(max_to_keep=5)
        saver.save(self.sess, model_checkpoint_filename, global_step=self.step_idx)

    def load(self, model_checkpoint_filename=None, var_scope=None):
        if model_checkpoint_filename is None:
            model_checkpoint_filename = os.path.join(self.model_checkpoint_dirname(), self.model_name)
        
        print('[*] Loading checkpoints from {}'.format(model_checkpoint_filename))

        if var_scope is not None:            
            variables_to_restore = [ v for v in slim.get_variables_to_restore(include=[var_scope]) if 'Adam' not in v.name ]
            init_fn = slim.assign_from_checkpoint_fn(model_checkpoint_filename, variables_to_restore, ignore_missing_vars=True)
            init_fn(self.sess)
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, model_checkpoint_filename)

    def build(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weight_decay = tf.placeholder(tf.float32, name='weight_decay')

    def build_summary(self):
        self.summary = tf.summary.merge_all()

        train_summary_dirname = os.path.join('./log', self.model_class(), self.model_name, 'train')
        self.train_writer = tf.summary.FileWriter(train_summary_dirname, self.sess.graph)

        valid_summary_dirname = os.path.join('./log', self.model_class(), self.model_name, 'valid')
        self.valid_writer = tf.summary.FileWriter(valid_summary_dirname)

