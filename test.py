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

flags.DEFINE_integer('batch_size', 128, '')

#flags.DEFINE_string('videos_dirname', None, '')
#flags.DEFINE_string('gt_filename', None, '')
#flags.DEFINE_string('result_dirname', None, '')
flags.DEFINE_string('model_name', 'small_lr-800', '')
flags.DEFINE_string('checkpoint_dirname', 'checkpoint', '')

FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config) as sess:

    # Initializaing and building model 
    model = ResNet50(
        sess=sess,
        model_name=FLAGS.model_name,
        checkpoint_dirname=FLAGS.checkpoint_dirname,
    )
    model.build(n_class=10)
    model.load()

    data_dirname = 'data/Fashion_MNIST_student/test'

    im_names, labels = my_IO.load_dataset(data_dirname, return_labels=False)
    N = len(im_names)

    dataset = my_IO.build_dataset(im_names, labels, n_epoch=1, batch_size=FLAGS.batch_size, shuffle=False)
    images, labels = dataset.make_one_shot_iterator().get_next()

    time_total = 0
    preds = []
    for start_idx in count(step=FLAGS.batch_size):
        try:
            b_images, _ = sess.run([images, labels])
        except tf.errors.OutOfRangeError:
            print()  # Done!
            break

        time_start = time.time()

        current_batch_size = len(b_images)
        end_idx = start_idx + current_batch_size

        b_preds = model.predict(b_images)
        preds.append(b_preds)

        b_time = int( time.time()-time_start )
        time_total += b_time
        time_eta = int( time_total * (N-end_idx) / end_idx )

        print('[Prediction] [ETA: {} s] [{:.2f} s/it]'.format(
                time_eta, 
                b_time,
            ), end='\r')

    print()
    preds = np.concatenate(preds, axis=0)
    my_IO.save_prediction('task_1.csv', im_names, preds)
    #np.savetxt('task_1.csv', preds, fmt='%d')