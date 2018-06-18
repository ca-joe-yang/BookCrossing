#! python3

import os
import sys
import my_IO
import evaluate

import numpy as np

from scipy import optimize

class Naive:

    def __init__(self):
        pass

    def fit(self, R):
        self.mu = R.sum() / R.nnz
        N, M = R.shape

        s = np.squeeze( np.array(R.sum(axis=1)) )
        n = R.getnnz(axis=1)
        mask = n>0
        self.d_users = np.full(N, self.mu)
        self.d_users[mask] = np.divide( s[mask], n[mask] )
        self.d_users = self.d_users - self.mu

        s = np.squeeze( np.array(R.sum(axis=0)) )
        m = R.getnnz(axis=0)
        mask = m>0
        self.d_books = np.full(M, self.mu)
        self.d_books[mask] = np.divide( s[mask], m[mask] )
        self.d_books = self.d_books - self.mu

    def predict(self, user_ids, book_ids):
        scores = self.mu + self.d_users[user_ids] + self.d_books[book_ids]
        scores[scores > 10] = 10
        scores[scores <  0] = 0
        return scores

class Baseline:

    def __init__(self):
        pass

    def fit(self, R, n_epoch=20, batch_size=8, lr=1e-2, lambda_=0.002):
        self.mu = R.sum() / R.nnz
        N, M = R.shape

        users, books = R.nonzero()
        self.bu = np.zeros(N)
        self.bi = np.zeros(M)

        def chunks(l, n):
            return [l[i:i + n] for i in range(0, len(l), n)] 

        U_batches = chunks(users, batch_size)
        I_batches = chunks(books, batch_size)

        for epoch_idx in range(n_epoch):
            count = 0
            losses = []
            for step_idx, (U_batch, I_batch) in enumerate(zip(U_batches, I_batches)):
                current_batch_size = len(U_batch)
                count += current_batch_size
                
                R_batch = np.array(R[U_batch, I_batch])[0]

                preds = self.mu + self.bu[U_batch] + self.bi[I_batch]
                err = R_batch - preds
                losses.append(np.average(np.abs(err)))

                self.bu[U_batch] += lr * (err - lambda_ * self.bu[U_batch])
                self.bi[I_batch] += lr * (err - lambda_ * self.bi[I_batch])

                loss_mean = np.average(losses)
                print('{}: {}/{}, loss:{:.3f}'.format(epoch_idx+1, count, len(users), loss_mean), end='\r')
            print()

    def predict(self, user_ids, book_ids):
        scores = self.mu + self.bu[user_ids] + self.bi[book_ids]
        scores[scores > 10] = 10
        scores[scores <  0] = 0
        return scores

users_name = np.genfromtxt('users_name.csv', dtype=str)
n_users = len(users_name)
books_ISBN = np.genfromtxt('books_ISBN.csv', dtype=str)
n_books = len(books_ISBN)
users_name2id = dict(zip(users_name, range(n_users)))
books_ISBN2id = dict(zip(books_ISBN, range(n_books)))

ratings_train = my_IO.read_ratings_train(users_name2id, books_ISBN2id, implicit=False)
test_user_ids, test_book_ids = my_IO.read_test(users_name2id, books_ISBN2id)

'''
model = Naive()
model.fit(ratings_train)
ans = model.predict(test_user_ids, test_book_ids)
np.savetxt('naive_wo_implicit.csv', ans.astype(int), fmt='%d')
'''

model = Baseline()
model.fit(ratings_train)
result = model.predict(test_user_ids, test_book_ids)
np.savetxt('baseline_wo_implicit.csv', result.astype(int), fmt='%d')
