import csv, sys
import numpy as np
import math
import random
from math import log, floor
from numpy.linalg import inv
import time
import sys



def normalize(X_all, X_test, dropnum):
    index = [112, 123, 1, 88, 102, 53, 39, 107, 121, 103, 5, 19, 110, 63, 118, 104, 117, 120, 89, 90,
             82, 95, 115, 122, 27, 84, 113, 99, 81, 98, 13, 83, 16, 48, 106, 94, 91, 108, 34, 109, 87,
             21, 24, 97, 101, 33, 111, 100, 85, 20, 119, 44, 29, 96, 93, 92, 71, 40, 114, 45, 28, 75,
             7, 26, 11, 25, 14, 54, 86, 65, 35, 30, 31, 55, 17, 74, 42, 58, 116, 46, 77, 57, 23, 52,
             59, 70, 51, 73, 15, 6, 60, 9, 2, 66, 38, 61, 64, 47, 12, 105, 72,
             41, 4, 36, 56, 3, 32, 8, 67, 18, 69, 37, 22, 62, 76, 79, 49, 50, 68, 78, 43, 80, 0, 10]
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    for i in [0,10,78,79,80]:

        mu = (sum(X_train_test[:,i]) / X_train_test.shape[0])
        sigma = np.std(X_train_test[:,i], axis=0)
        X_train_test[:,i] = (X_train_test[:,i] - mu) / sigma

        X_train_test = np.concatenate((X_train_test,(X_train_test[:,i]**2).reshape((X_train_test.shape[0],1))),axis=1)
        X_train_test = np.concatenate((X_train_test,(X_train_test[:,i]**3).reshape((X_train_test.shape[0],1))),axis=1)
        X_train_test = np.concatenate((X_train_test,(X_train_test[:,i]**4).reshape((X_train_test.shape[0],1))),axis=1)
        X_train_test = np.concatenate((X_train_test,(X_train_test[:,i]**5).reshape((X_train_test.shape[0],1))),axis=1)
        X_train_test = np.concatenate((X_train_test,(X_train_test[:,i]**6).reshape((X_train_test.shape[0],1))),axis=1)
        np.concatenate((X_train_test,(np.log(X_train_test[:,i]+1)).reshape((X_train_test.shape[0],1))),axis=1)

    X_train_test_normed = X_train_test

    X_all = X_train_test_normed[:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def load_data(x_train,y_train,t_train,v_size=1,rand=False,split=0,block=0,norm=True,bias=True,selected=False,dropnum=0):
    X = None
    Y = None
    Vx = None
    Vy = None
    if not selected:
        train_X = open(x_train)
        train_Y = open(y_train)
        X = list(csv.reader(train_X))
        title, X = X[0], X[1:]
        Y = list(csv.reader(train_Y))
        X = np.array(X, dtype=float)
        np.save('X.npy', X)
        np.save('Y.npy', np.array(Y, dtype=float).flatten())
        X = np.load('X.npy')
        test_X = open(t_train)
        T = list(csv.reader(test_X))
        title2, T = T[0], T[1:]
        T = np.array(T, dtype=float)
        np.save('T.npy', T)
        T = np.load('T.npy')
    else:
        X = np.load('new_X.npy')
        T = np.load('new_T.npy')

    Y = np.load('Y.npy')

    if norm:
        X, T = normalize(X, T,dropnum)
    if bias:
        T = np.concatenate((T, np.ones((T.shape[0], 1))), axis=1)
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    if rand:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X,Y = (X[randomize], Y[randomize])
    v_size = floor(len(X) * v_size)
    Vx = X[v_size:] if v_size != len(X) else X[floor(len(X) * 0.9):]
    Vy = Y[v_size:] if v_size != len(X) else Y[floor(len(X) * 0.9):]
    X = X[:v_size]
    Y = Y[:v_size]

    if split!=0:
        v_size = floor(len(X) // split)
        block-=1
        if block not in range(split):
            block = 0
            print("FUCK!! please choose the right block!")
        Vx = X[block*v_size:(block+1)*v_size]
        Vy = Y[block*v_size:(block+1)*v_size]
        X = np.concatenate((X[:block*v_size],X[(block+1)*v_size:]))
        Y = np.concatenate((Y[:block*v_size],Y[(block+1)*v_size:]))
    return X, Y, Vx, Vy, T

def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-12, 1-(1e-12))

if __name__=='__main__':
    X, Y, Vx, Vy, T = load_data(sys.argv[1],sys.argv[2],sys.argv[3],v_size=1,rand=False,dropnum=0,bias=True)
    w = np.load('model_w_Dis.npy')
    idx = 0
    ans = []

    for i in T:
        x = np.array(i, dtype=float)
        res = sigmoid(w.dot(x))
        if res >= 0.5:
            res = 1
        else:
            res = 0
        idx += 1
        ans.append([idx, res])
    filename = sys.argv[4]
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()