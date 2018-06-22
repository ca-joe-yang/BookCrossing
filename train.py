#! python3

import os
import sys
import time
import my_IO

import tensorflow as tf
import numpy as np

from model import *
from itertools import count

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

flags = tf.app.flags

flags.DEFINE_integer('n_epoch', 20, 'Epochs to train [50]')
flags.DEFINE_integer('batch_size', 8, '')

flags.DEFINE_float('lr', 1e-3, 'Learning rate [0.00017]')
#flags.DEFINE_float('keep_prob', 0.5, '')
#flags.DEFINE_float('weight_decay', 0.0005, '')

flags.DEFINE_boolean('restore_ckpt', False, '')

#flags.DEFINE_string('data_dirname', 'HW5_data', '')
flags.DEFINE_string('model_name', 'baseline', '')
flags.DEFINE_string('checkpoint_dirname', 'checkpoint', '')

FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

users_name = np.genfromtxt('users_name.csv', dtype=str)
n_users = len(users_name)
books_ISBN = np.genfromtxt('books_ISBN.csv', dtype=str)
n_books = len(books_ISBN)
users_name2id = dict(zip(users_name, range(n_users)))
books_ISBN2id = dict(zip(books_ISBN, range(n_books)))

R_train = my_IO.read_ratings_train(users_name2id, books_ISBN2id, implicit=False)
mu = R_train.sum() / R_train.nnz

with tf.Session(config=config) as sess:

    # Initializaing and building model 
    model = Baseline(
        sess=sess,
        model_name=FLAGS.model_name,
        checkpoint_dirname=FLAGS.checkpoint_dirname,
    )
    model.build(mu=mu, N=n_users, M=n_books)
    model.build_loss()
    model.build_optimizer(optimizer='adam', lr=FLAGS.lr)
    model.build_summary()
    
    model.initialize()
    if FLAGS.restore_ckpt:
        model.load()

    dataset = my_IO.build_dataset(R_train, n_epoch=FLAGS.n_epoch, batch_size=FLAGS.batch_size, shuffle=True)
    user_ids, item_ids, labels = dataset.make_one_shot_iterator().get_next()

    #best_evaluation = -np.float('inf')
    time_total = 0
    max_iter = int(FLAGS.n_epoch * R_train.nnz / FLAGS.batch_size)
    for start_idx in count(step=FLAGS.batch_size):
        try:
            b_user_ids, b_item_ids, b_labels = sess.run([user_ids, item_ids, labels])
        except tf.errors.OutOfRangeError:
            print()  # Done!
            break

        time_start = time.time()

        current_batch_size = len(b_user_ids)
        end_idx = start_idx + current_batch_size

        b_loss, b_acc = model.train_valid(b_user_ids, b_item_ids, b_labels)

        b_time = int( time.time()-time_start )
        time_total += b_time
        time_eta = int( time_total * (max_iter-model.step_idx) / model.step_idx )

        print('\x1b[2K[Train] [Step: {}] [ETA: {} s] [{:.4f} s/it] '
              '[Loss: {:.3f}] [Acc: {:.3f}]'.format(
                model.step_idx, 
                time_eta, 
                b_time,
                np.average(b_loss), 
                np.average(b_acc)
            ), end='\r')

        if model.step_idx % 100 == 0:
            print()
            model.save()

        '''
        valid_preds = model.predict(valud_im_names, FLAGS.batch_size)
        valid_acc = np.mean( np.equal(valid_preds, valid_labels) )
        print('\x1b[2K\t[Valid] [Acc: {:.3f}]'.format(valid_acc))
       
        if valid_acc > best_evaluation:
            best_evaluation = valid_acc
            model.save()
        '''
