#! python3

import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from scipy import sparse
'''
0: ISBN
1: Book-Title
2: Book-Author
3: Year-Of-Publication
4: Publisher
5: Image-URL-S
6: Image-URL-M
    - {%10d}.01.MZZZZZZZ.jpg
7: Image-URL-L
8: Book-Descript

Users: 278858
Books: 271379
Rating: 260202
<278860x271381 sparse matrix of type '<class 'numpy.int64'>'
    with 243402
'''

def build_dataset(R, n_epoch, batch_size, shuffle):
    user_ids, book_ids = R.nonzero()
    labels = np.array(R[user_ids, book_ids])[0]

    dataset = tf.data.Dataset.from_tensor_slices((user_ids, book_ids, labels))

    if shuffle:
        dataset = dataset.shuffle(5000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(n_epoch)

    return dataset


def check_ISBN_format(x):
    return re.sub('[^A-Za-z0-9]', '', x)

def get_books_ISBN(filename):
    books = np.array(pd.read_csv(filename, header=0)['ISBN']).astype(str)
    books = [ check_ISBN_format(b) for b in books ]
    return books

def get_users_name(filename):
    users = np.array(pd.read_csv(filename, header=0)['User-ID']).astype(str)
    return users

def get_user_ages(filename):
    ages = np.array(pd.read_csv(filename, header=0)['Age']).astype(int)
    avg = np.sum(ages[ages > 0]) / np.sum(ages > 0)
    ages[ages < 0] = avg
    return ages

def get_ratings(filename):
    R = np.array(pd.read_csv('data/book_ratings_train.csv', header=0)['Book-Rating'])
    #_ = np.histogram(ratings, 9)

def save_all_users_name():
    users = get_users_name('data/users.csv')
    users.sort()
    np.savetxt('users_name.csv', users, fmt='%s')

def save_all_books_ISBN():
    books_1 = get_books_ISBN('data/books.csv')
    books_2 = get_books_ISBN('data/book_ratings_train.csv')
    books_3 = get_books_ISBN('data/book_ratings_test.csv')
    books_4 = get_books_ISBN('data/implicit_ratings.csv')

    print('Books in "book_ratings_train.csv" but not in "books.csv": {}'.format(
            len(np.setdiff1d(books_2, books_1))
        ))
    print('Books in "book_ratings_test.csv" but not in "books.csv": {}'.format(
            len(np.setdiff1d(books_3, books_1))
        ))
    print('Books in "implicit_ratings.csv" but not in "books.csv": {}'.format(
            len(np.setdiff1d(books_4, books_1))
        ))
    print('Books in "book_ratings_test.csv" but not in "book_ratings_train.csv": {}'.format(
            len(np.setdiff1d(books_3, books_2))
        ))

    books = set(books_1) | set(books_2) | set(books_3) | set(books_4)
    books = list(books)
    books.sort()
    np.savetxt('books_ISBN.csv', books, fmt='%s')

def test():
    x = pd.read_csv('data/users.csv', header=0)
    y = x.set_index('User-ID').T.to_dict('list')

    books = pd.read_csv('data/books.csv', header=0)['ISBN']
    books = np.array(books).astype(str)

def get_countries():
    data = pd.read_csv('data/users.csv', header=0)
    locations = np.array( data['Location'].astype(str) )
    countries = [ l.split(',')[-1] for l in locations ]

def read_test(users_name2id=None, books_ISBN2id=None):
    
    filename = 'data/book_ratings_test.csv'
    user_ids = []
    book_ids = []
    with open(filename, 'r') as f:
        data_reader = csv.reader(f)

        for i, x in enumerate(data_reader):
            if i == 0:
                continue
            print('{}: {}'.format(filename, i), end='\r')
            user_id = users_name2id[x[0]]
            user_ids.append(user_id)
            book_id = books_ISBN2id[check_ISBN_format(x[1])]
            book_ids.append(book_id)
    print()
    return np.array(user_ids), np.array(book_ids)

def read_ratings_train(users_name2id=None, books_ISBN2id=None, implicit=False, split=0):

    N = len(users_name2id)
    M = len(books_ISBN2id)

    if implicit:
        npy_filename = 'ratings_train_all.npy'
    else:
        npy_filename = 'ratings_train_nonzero.npy'

    user_ids = []
    book_ids = []
    ratings = []

    filename1 = 'data/book_ratings_train.csv'
    with open(filename1, 'r') as f:
        data_reader = csv.reader(f)

        for i, x in enumerate(data_reader):
            if i == 0:
                continue
            print('{}: {}'.format(filename1, i), end='\r')
            user_id = users_name2id[x[0]]
            user_ids.append(user_id)
            book_id = books_ISBN2id[check_ISBN_format(x[1])]
            book_ids.append(book_id)
            ratings.append(int(x[2]))
    print()

    if implicit:
        filename2 = 'data/implicit_ratings.csv'
        with open(filename2, 'r') as f:
            data_reader = csv.reader(f)

            for i, x in enumerate(data_reader):
                if i == 0:
                    continue
                print('{}: {}'.format(filename2, i), end='\r')
                user_id = users_name2id[x[2]]
                user_ids.append(user_id)
                book_id = books_ISBN2id[check_ISBN_format(x[1])]
                book_ids.append(book_id)
                ratings.append(int(x[0]))
        print()
    
    R_train = sparse.csc_matrix( (ratings[split:], (user_ids[split:], book_ids[split:])), shape=[N,M] )
    R_valid = sparse.csc_matrix( (ratings[:split], (user_ids[:split], book_ids[:split])), shape=[N,M] )

    return R_train, R_valid

def read_users(data_dirname='data'):
    users_filename = os.path.join(data_dirname, 'users.csv')
    users = np.genfromtxt(users_filename, skip_header=1)
    users_name = users[:, 0]
    users_name.sort()
    N = users.shape[0]
    users_name2id = dict( users_name, range(N) )
    pass

def download_books_cover_images(data_dirname='data', size='L'):
    assert size in ['S', 'M', 'L']

    im_dirname = os.path.join(data_dirname, 'images')
    if not os.path.exists(im_dirname):
        os.mkdir(im_dirname)
    
    if size == 'S':
        column = 5
        im_name_format = '{}.01.THUMBZZZ.jpg'
    elif size == 'M':
        column = 6
        im_name_format = '{}.01.MZZZZZZZ.jpg'
    elif size == 'L':
        column = 7
        im_name_format = '{}.01.LZZZZZZZ.jpg'
    
    books_filename = os.path.join(data_dirname, 'books.csv')
    with open(books_filename, 'r') as f:
        data_reader = csv.reader(f)

        for i, x in enumerate(data_reader):
            if i == 0:
                continue
            url = x[column]
            ISBN = x[0]
            im_name = im_name_format.format(ISBN)
            #print('Books {}'.format(i+1), end='\r')
            if not os.path.exists(im_name):
                wget.download(url, out=im_dirname)
    print()


if __name__ == '__main__':
    #X = download_books_cover_images()
    #X = read_book_crossing()
    #user_ids, book_ids = read_book_crossing_test(X=X)
    '''
    location_corpus = np.array(pd.read_csv('data/users.csv', header=0)['Location']).astype(str)
    tmp = []
    for line in location_corpus:
        line = re.sub('[^,\\ A-Za-z]', '', line)
        tokens = line.split(',')
        tokens = [ '-'.join(t.strip().split()) for t in tokens ]
        line = ' , '.join(tokens)
        tmp.append(line)
    location_corpus = '\n'.join(tmp)
    with open('location_corpus.txt', 'w') as f:
        f.write(location_corpus)
    '''
    #users = get_users_name('data/users.csv')

    save_all_books_ISBN()

    pass