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

flags.DEFINE_integer('batch_size', 32, '')

flags.DEFINE_string('model_name', 'new-121980', '')
flags.DEFINE_string('checkpoint_dirname', 'checkpoint', '')

FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

user_names = np.genfromtxt('users_name.csv', dtype=str)
user_embeds = my_IO.get_user_embeds('data/users.csv')
user_embeds = np.array([ user_embeds[name] for name in user_names ])
n_users = len(user_names)
book_ISBNs = np.genfromtxt('books_ISBN.csv', dtype=str)
n_books = len(book_ISBNs)
user_name2id = dict(zip(user_names, range(n_users)))
book_ISBN2id = dict(zip(book_ISBNs, range(n_books)))

R_train, _ = my_IO.read_ratings_train(user_name2id, book_ISBN2id, implicit=False)
mu = R_train.sum() / R_train.nnz

with tf.Session(config=config) as sess:

    # Initializaing and building model 
    model = Baseline(
        sess=sess,
        model_name=FLAGS.model_name,
        checkpoint_dirname=FLAGS.checkpoint_dirname,
    )
    model.build(mu=mu, N=n_users, M=n_books)
    model.load()

    test_user_ids, test_book_ids = my_IO.read_test(user_name2id, book_ISBN2id)
    test_user_embeds = user_embeds[test_user_ids].reshape(-1, 1)

    result = model.predict(test_user_ids, test_book_ids, FLAGS.batch_size)
    np.savetxt('latent.csv', result.astype(int), fmt='%d')
