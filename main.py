#! python3

import os
import sys
import my_IO
import my_util
import evaluate

import numpy as np

from scipy import optimize

class Naive:

    def __init__(self, N, M):
        # overall average
        self.mu = 0

        # observed deviations of user u
        self.bu = np.zeros(N)

        # observed deviations of item i
        self.bi = np.zeros(M)

    def fit(self, R):
        self.mu = R.sum() / R.nnz

        # compute observed deviations of user u        
        s = np.squeeze( np.array(R.sum(axis=1)) )
        n = R.getnnz(axis=1)
        mask = n>0
        self.bu[mask] = np.divide( s[mask], n[mask] ) - self.mu

        # compute observed deviations of item i        
        s = np.squeeze( np.array(R.sum(axis=0)) )
        m = R.getnnz(axis=0)
        mask = m>0
        self.bi[mask] = np.divide( s[mask], m[mask] ) - self.mu

    def predict(self, user_ids, book_ids):
        scores = self.mu + self.bu[user_ids] + self.bi[book_ids]
        scores[scores > 10] = 10
        scores[scores <  0] = 0
        return scores

    def save(self):
        np.save('naive.npy', [self.mu, self.bu, self.bi])

    def load(self):
        self.mu, self.bu, self.bi = np.load('naive.npy')

class Baseline:

    def __init__(self, N, M):
        # overall average
        self.mu = 0

        # observed deviations of user u
        self.bu = np.zeros(N)

        # observed deviations of item i
        self.bi = np.zeros(M)

    def fit(self, R, n_epoch=40, batch_size=8, lr=1e-2, lambda_=0.002):
        self.mu = R.sum() / R.nnz

        users, books = R.nonzero()
        U_batches = my_util.chunks(users, batch_size)
        I_batches = my_util.chunks(books, batch_size)

        for epoch_idx in range(n_epoch):
            count = 0
            losses = []

            U_batches, I_batches = my_util.shuffle_xy(U_batches, I_batches)

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

    def save(self):
        np.save('baseline.npy', [self.mu, self.bu, self.bi])

    def load(self):
        self.mu, self.bu, self.bi = np.load('baseline.npy')

class LatentFactor:

    def __init__(self, N, M, latent_dim=20):
        # overall average
        self.mu = 0

        # observed deviations of user u
        self.bu = np.zeros(N)

        # observed deviations of item i
        self.bi = np.zeros(M)
        
        # user-factors vector
        self.p = np.zeros([N, latent_dim]) + 0.1

        # item-factors vector
        self.q = np.zeros([M, latent_dim]) + 0.1

    def fit(self, R, n_epoch=20, batch_size=8, lr=1e-2, lambda_=0.005):
        self.mu = R.sum() / R.nnz

        users, books = R.nonzero()
        U_batches = my_util.chunks(users, batch_size)
        I_batches = my_util.chunks(books, batch_size)

        for epoch_idx in range(n_epoch):
            count = 0
            losses = []
            loss_mean = 0

            U_batches, I_batches = my_util.shuffle_xy(U_batches, I_batches)

            for step_idx, (U_batch, I_batch) in enumerate(zip(U_batches, I_batches)):
                current_batch_size = len(U_batch)
                count += current_batch_size
                
                labels = np.array(R[U_batch, I_batch])[0]

                preds = self.mu + self.bu[U_batch] + self.bi[I_batch] + np.einsum('ij,ij->i', self.p[U_batch], self.q[I_batch])
                err = labels - preds
                losses.append(np.average(np.abs(err)))

                self.bu[U_batch] += lr * (err - lambda_ * self.bu[U_batch])
                self.bi[I_batch] += lr * (err - lambda_ * self.bi[I_batch])
                self.p[U_batch] += lr * ([ a*b for a,b in zip(err, self.q[I_batch]) ] - lambda_ * self.p[U_batch])
                self.q[I_batch] += lr * ([ a*b for a,b in zip(err, self.p[U_batch]) ] - lambda_ * self.q[I_batch])

                if count % 10000 == 0:
                    loss_mean = np.average(losses)
                print('{}: {}/{}, loss:{:.3f}'.format(epoch_idx+1, count, len(users), loss_mean), end='\r')
            print()

    def predict(self, user_ids, book_ids):
        scores = self.mu + self.bu[user_ids] + self.bi[book_ids] + np.einsum('ij,ij->i', self.p[user_ids], self.q[book_ids])
        scores[scores > 10] = 10
        scores[scores <  0] = 0
        return scores

    def save(self):
        np.save('latent_factor.npy', [self.mu, self.bu, self.bi, self.p, self.q])

    def load(self):
        self.mu, self.p, self.q, self.bu, self.bi = np.load('latent_factor.npy')


users_name = np.genfromtxt('users_name.csv', dtype=str)
n_users = len(users_name)
books_ISBN = np.genfromtxt('books_ISBN.csv', dtype=str)
n_books = len(books_ISBN)
users_name2id = dict(zip(users_name, range(n_users)))
books_ISBN2id = dict(zip(books_ISBN, range(n_books)))

ratings_train = my_IO.read_ratings_train(users_name2id, books_ISBN2id, implicit=False)
test_user_ids, test_book_ids = my_IO.read_test(users_name2id, books_ISBN2id)

'''
model = Naive(N=n_users, M=n_books)
model.fit(ratings_train)
result = model.predict(test_user_ids, test_book_ids)
np.savetxt('naive_wo_implicit.csv', result.astype(int), fmt='%d')
'''

'''
model = Baseline(N=n_users, M=n_books)
model.fit(ratings_train)
result = model.predict(test_user_ids, test_book_ids)
np.savetxt('baseline_wo_implicit.csv', result.astype(int), fmt='%d')
'''

model = LatentFactor(N=n_users, M=n_books, latent_dim=10)
model.fit(ratings_train, 10)
result = model.predict(test_user_ids, test_book_ids)
np.savetxt('latent_wo_implicit.csv', np.rint(result), fmt='%d')
