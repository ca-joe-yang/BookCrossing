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

flags.DEFINE_string('model_name', 'mse-26429', '')
flags.DEFINE_string('checkpoint_dirname', 'checkpoint', '')

FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

users_name = np.genfromtxt('users_name.csv', dtype=str)
user_ages = my_IO.get_user_ages('data/users.csv') / 100.0
n_users = len(users_name)
books_ISBN = np.genfromtxt('books_ISBN.csv', dtype=str)
n_books = len(books_ISBN)
users_name2id = dict(zip(users_name, range(n_users)))
books_ISBN2id = dict(zip(books_ISBN, range(n_books)))

R_train, _ = my_IO.read_ratings_train(users_name2id, books_ISBN2id, implicit=False)
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

    test_user_ids, test_book_ids = my_IO.read_test(users_name2id, books_ISBN2id)
    test_user_ages = user_ages[test_user_ids].reshape(-1, 1)

    result = model.predict(test_user_ids, test_book_ids, FLAGS.batch_size)
    np.savetxt('latent.csv', result.astype(int), fmt='%d')
