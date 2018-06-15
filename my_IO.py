#! python3

import os
import csv
import wget
import numpy as np
import pandas as pd
import re
from scipy import sparse
from glove import Glove

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

def get_books_ISBN(filename):
    books = np.array(pd.read_csv(filename, header=0)['ISBN']).astype(str)
    books = [ re.sub('[^A-Za-z0-9]', '', b) for b in books ]
    return books

def save_all_books_ISBN(filename):
    books_1 = get_books_ISBN('data/books.csv')
    books_2 = get_books_ISBN('data/book_ratings_train.csv')
    books_3 = get_books_ISBN('data/book_ratings_test.csv')
    books_4 = get_books_ISBN('data/implicit_ratings.csv')

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

def read_book_crossing_test(data_dirname='data', users_name2id=None, books_ISBN2id=None):
    
    rating_filename = os.path.join(data_dirname, 'book_ratings_test.csv')
    user_ids = []
    book_ids = []
    with open(rating_filename, 'r') as f:
        data_reader = csv.reader(f)

        for i, x in enumerate(data_reader):
            if i == 0:
                continue
            print('Rating: {}'.format(i), end='\r')
            if x[0] not in users_name2id:
                user_id = users_name2id['unknown_user']
                #print('User {} unknown'.format(x[0]))
            else:
                user_id = users_name2id[x[0]]
            user_ids.append(user_id)
            if x[1] not in books_ISBN2id:
                book_id = books_ISBN2id['unknown_book']
                #print('Book {} unknown'.format(x[1]))
            else:
                book_id = books_ISBN2id[x[1]]
            book_ids.append(book_id)
    return np.array(user_ids), np.array(book_ids)

def read_book_crossing(data_dirname='data'):
    users_filename = os.path.join(data_dirname, 'users.csv')
    users_name = ['unknown_user']
    with open(users_filename, 'r') as f:
        data_reader = csv.reader(f)

        for i, x in enumerate(data_reader):
            if i == 0:
                continue
            print('Users: {}'.format(i), end='\r')
            users_name.append(x[0])
    print()
    
    users_name.sort()
    N = len(users_name)
    users_name2id = dict( zip(users_name, range(N)) )

    books_filename = os.path.join(data_dirname, 'books.csv')
    books_ISBN = ['unknown_book']
    with open(books_filename, 'r') as f:
        data_reader = csv.reader(f)

        for i, x in enumerate(data_reader):
            if i == 0:
                continue
            print('Books: {}'.format(i), end='\r')
            ISBN = x[0]
            ISBN.replace('.', '')
            books_ISBN.append(ISBN)
    print()

    books_ISBN.sort()
    M = len(books_ISBN)
    books_ISBN2id = dict( zip(books_ISBN, range(M)) )

    if not os.path.exists('ratings_train.npy'):
        rating_filename = os.path.join(data_dirname, 'book_ratings_train.csv')
        #rating_matrix = np.zeros([N, M])
        user_ids = []
        book_ids = []
        ratings = []
        with open(rating_filename, 'r') as f:
            data_reader = csv.reader(f)

            for i, x in enumerate(data_reader):
                if i == 0:
                    continue
                print('Rating: {}'.format(i), end='\r')
                if x[0] not in users_name:
                    user_id = users_name2id['unknown_user']
                    #print('User {} unknown'.format(x[0]))
                else:
                    user_id = users_name2id[x[0]]
                user_ids.append(user_id)
                if x[1] not in books_ISBN:
                    book_id = books_ISBN2id['unknown_book']
                    #print('Book {} unknown'.format(x[1]))
                else:
                    book_id = books_ISBN2id[x[1]]
                book_ids.append(book_id)
                
                ratings.append(int(x[2]))
        print()
        rating_matrix = sparse.csc_matrix( (ratings, (user_ids,book_ids)), shape=[N+1,M+1] )
        np.save('ratings_train.npy', rating_matrix)
    rating_matrix = np.load('ratings_train.npy')[()]

    data = dict(
        N = N,
        M = M,
        users_name2id = users_name2id,
        books_ISBN2id = books_ISBN2id,
        rating_matrix = rating_matrix,
    )

    return data

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

    pass