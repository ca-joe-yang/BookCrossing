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
flags.DEFINE_integer('batch_size', 128, '')

flags.DEFINE_float('lr', 1e-5, 'Learning rate [0.00017]')
#flags.DEFINE_float('keep_prob', 0.5, '')
flags.DEFINE_float('weight_decay', 0.000005, '')

flags.DEFINE_boolean('restore_ckpt', True, '')

flags.DEFINE_string('model_name', 'small_bs-40660-10165', '')
flags.DEFINE_string('checkpoint_dirname', 'checkpoint', '')

FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

user_names = np.genfromtxt('users_name.csv', dtype=str)
n_users = len(user_names)

book_ISBNs = np.genfromtxt('books_ISBN.csv', dtype=str)
item_embeds = my_IO.get_item_embeds('books_ignore_oov.npy', item_names=book_ISBNs)
n_books = len(book_ISBNs)

user_name2id = dict(zip(user_names, range(n_users)))
book_ISBN2id = dict(zip(book_ISBNs, range(n_books)))

user_embeds = my_IO.get_user_embeds('data/users.csv', item_name2id=book_ISBN2id, item_embeds=item_embeds)
user_embeds = np.array([ user_embeds[name] for name in user_names ])
print(user_embeds.shape)

split = 0
R_train, R_valid = my_IO.read_ratings_train(user_name2id, book_ISBN2id, implicit=False, split=split)
mu = R_train.sum() / R_train.nnz

with tf.Session(config=config) as sess:
    result_filename = 'latent.csv'
    # Initializaing and building model 
    model = Embed(
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

    train_dataset = my_IO.build_dataset(R_train, n_epoch=FLAGS.n_epoch, batch_size=FLAGS.batch_size, shuffle=True)
    train_user_ids, train_item_ids, train_labels = train_dataset.make_one_shot_iterator().get_next()
    n_train = R_train.nnz
    n_train_batch = np.ceil(n_train / FLAGS.batch_size).astype(int)
    

    if split > 0:
        valid_dataset = my_IO.build_dataset(R_train, n_epoch=FLAGS.n_epoch, batch_size=FLAGS.batch_size, shuffle=False)
        valid_user_ids, valid_item_ids, valid_labels = valid_dataset.make_one_shot_iterator().get_next()
        n_valid = R_valid.nnz
        n_valid_batch = np.ceil(n_valid / FLAGS.batch_size).astype(int)    

    for epoch_idx in range(FLAGS.n_epoch):
        train_loss = []
        train_acc = []
        count = 0
        time_total = 0
        for _ in range(n_train_batch):
            b_user_ids, b_item_ids, b_labels = sess.run([train_user_ids, train_item_ids, train_labels])
            b_user_embeds = user_embeds[b_user_ids]
            b_item_embeds = item_embeds[b_item_ids]

            time_start = time.time()

            current_batch_size = len(b_user_ids)

            b_loss, b_acc = model.train_valid(b_user_ids, b_item_ids, b_user_embeds, b_item_embeds, b_labels, is_training=True, weight_decay=FLAGS.weight_decay)

            b_time = time.time()-time_start
            count += current_batch_size
            time_total += b_time
            time_eta = int( time_total * (n_train / count - 1))

            train_loss.append(b_loss)
            train_acc.append(b_acc)

            print('\x1b[2K[Train] [Step: {}] [ETA: {} s] [{:.4f} s/it] '
                  '[Loss: {:.3f}] [Acc: {:.3f}]'.format(
                    model.step_idx, 
                    time_eta, 
                    b_time,
                    np.average(train_loss), 
                    np.average(train_acc)
                ), end='\r')

        print()

        if split > 0:
            valid_loss = []
            valid_acc = []
            count = 0
            for _ in range(n_valid_batch):
                b_user_ids, b_item_ids, b_labels = sess.run([valid_user_ids, valid_item_ids, valid_labels])
                b_user_embeds = user_embeds[b_user_ids]
                b_item_embeds = item_embeds[b_item_ids]

                b_loss, b_acc = model.train_valid(b_user_ids, b_item_ids, b_user_embeds, b_item_embeds, b_labels, is_training=False)

                valid_loss.append(b_loss)
                valid_acc.append(b_acc)

                print('\x1b[2K[Valid] [Loss: {:.3f}] [Acc: {:.3f}]'.format(
                        np.average(valid_loss), 
                        np.average(valid_acc)
                    ), end='\r')
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

    test_user_ids, test_item_ids = my_IO.read_test(user_name2id, book_ISBN2id)
    test_user_embeds = user_embeds[test_user_ids]
    test_item_embeds = item_embeds[test_item_ids]
    result = model.predict(test_user_ids, test_item_ids, test_user_embeds, test_item_embeds, FLAGS.batch_size)
    result[result > 10] = 10
    result[result < 0] = 0
    np.savetxt(result_filename, np.rint(result), fmt='%d')

