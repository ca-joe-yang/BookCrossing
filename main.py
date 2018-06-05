#! python3

import numpy as np
import sys, os
import my_IO
import evaluate

def average_all(R):
    return R.sum() / (R>0).sum()

def deviation_users(R):
    mu = average_all(R)
    N = R.shape[0]
    a = np.squeeze( np.array(R.sum(axis=1)) )
    b = np.squeeze( np.array((R>0).sum(axis=1)) )
    mask = a>0
    d = np.full(N, mu)
    d[mask] = np.divide( a[mask], b[mask] )
    return d - mu

def deviation_books(R):
    mu = average_all(R)
    M = R.shape[1]
    a = np.squeeze( np.array(R.sum(axis=0)) )
    b = np.squeeze( np.array((R>0).sum(axis=0)) )
    mask = a>0
    d = np.full(M, mu)
    d[mask] = np.divide( a[mask], b[mask] )
    return d - mu

def predict(R, user_ids, book_ids):
    mu = average_all(R)
    d_users = deviation_users(R)
    d_books = deviation_books(R)
    scores = mu + d_users[user_ids] + d_books[book_ids]
    scores[scores > 10] = 10
    scores[scores <  0] = 0
    return scores.astype(int)

train_data = my_IO.read_book_crossing('data')
test_user_ids, test_book_ids = my_IO.read_book_crossing_test(
    data_dirname='data', users_name2id=train_data['users_name2id'], books_ISBN2id=train_data['books_ISBN2id'])
R = train_data['rating_matrix']

ans = predict(R, test_user_ids, test_book_ids)
np.savetxt('result.csv', ans, fmt='%d')