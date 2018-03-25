import csv, sys
import numpy as np
import math
import random
from math import log, floor
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    # X_train_test = np.delete(X_train_test,116,1)
    # X_train_test = np.delete(X_train_test,54,1)
    # X_train_test = np.delete(X_train_test,np.s_[27:43],1)
    # X_train_test = np.delete(X_train_test,7,1)
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def load_data(v_size=1,rand=True,split=0,block=0,norm=True,bias=True):
    train_X = open('train_X')
    train_Y = open('train_Y')
    X = list(csv.reader(train_X))
    title, X = X[0],X[1:]
    Y = list(csv.reader(train_Y))
    X = np.array(X, dtype=float)
    #cg = X.T[62]
    #np.insert(X,63,np.log(cg+1),axis=1)
    np.save('X.npy', X)
    np.save('Y.npy', np.array(Y, dtype=float).flatten())
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    test_X = open('test_X')
    T = list(csv.reader(test_X))
    title2, T = T[0], T[1:]
    T = np.array(T, dtype=float)
    #cg = T.T[62]
    #np.insert(T,63,np.log(cg+1),axis=1)
    np.save('T.npy', T)
    T = np.load('T.npy')
    if norm:
        X, T = normalize(X, T)
    if bias:
        T = np.concatenate((T, np.ones((T.shape[0], 1))), axis=1)
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    if rand:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X,Y = (X[randomize], Y[randomize])
    v_size = floor(len(X) * v_size)
    Vx = X[v_size:]
    Vy = Y[v_size:]
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
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-9, 1-(1e-9))
    # return 1 / (1.0 + np.exp(-z))

def save_gen_model(sigma,mu1,mu2, num1, num2):
    np.save("sigma.npy",sigma)
    np.save("mu1.npy",mu1)
    np.save("mu2.npy", mu2)
    np.save("num1.npy", num1)
    np.save("num2.npy", num2)

def load_gen_model():
    sigma = np.load("sigma.npy")
    mu1 = np.load("mu1.npy")
    mu2 = np.load("mu2.npy")
    num1 = np.load("num1.npy")
    num2 = np.load("num2.npy")
    return sigma,mu1,mu2, num1, num2

def validGen(Vx, Vy):
    sigma, mu1, mu2, num1, num2 = load_gen_model()
    #print(sigma)
    sigma_inv = np.linalg.inv(sigma)
    w = np.matmul((mu1 - mu2), sigma_inv)
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inv), mu1) + \
    (0.5) * np.dot(np.dot([mu2], sigma_inv), mu2) + np.log(
        float(num1) / num2)
    ans = sigmoid(w.dot(Vx.T)+b)
    succ = 0
    idx =0
    fail = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for res,j in zip(ans, Vy):
        #print(1.0-float(res),j)
        if res >= 0.5:
            res = 0
        else:
            res = 1
        if res == j:
            succ += 1
        else:
            fail += 1
        if j == 1:
            if res == 1:
                TP += 1
            else:
                FN += 1
        else:
            if res == 1:
                FP += 1
            else:
                TN += 1
        idx += 1
    print("score:", (succ / idx), "F1-measure:", (2 * TP / (2 * TP + FP + FN)))

def Gen(v_size=0.9,rand=True,split=0,block=0,norm=True):
    X, Y, Vx, Vy, T = load_data(v_size,rand,split,block,norm,bias=False)
    mu1 = np.zeros((X.shape[1],))
    mu2 = np.zeros((X.shape[1],))
    sigma1 = np.zeros((X.shape[1], X.shape[1]))
    sigma2 = np.zeros((X.shape[1], X.shape[1]))
    num1 = 0
    num2 = 0
    for x, y in zip(X,Y):
        if y==0:
            num1+=1
            mu1+=x
        else:
            num2+=1
            mu2 += x
    mu1/=num1
    mu2/=num2
    #print(mu1)
    for x, y in zip(X,Y):
        if y==0:
            sigma1+=np.matmul(np.transpose([x-mu1]),([x-mu1]))
        else:
            sigma2 +=np.matmul(np.transpose([x-mu2]),([x-mu2]))
    sigma1/=num1
    sigma2/=num2
    sigma = (num1/(num1+num2))*sigma1 + (num2/(num1+num2))*sigma2
    save_gen_model(sigma, mu1, mu2, num1, num2)
    validGen(Vx,Vy)

def testGen():
    _x,_y,_vx,_vy, X = load_data(bias=False)
    sigma, mu1, mu2, num1, num2 = load_gen_model()
    idx = 0
    ans = []
    sigma_inv = np.linalg.inv(sigma)
    w = np.matmul((mu1 - mu2), sigma_inv)
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inv), mu1) + \
        (0.5) * np.dot(np.dot([mu2], sigma_inv), mu2) + np.log(
        float(num1) / num2)
    sol = sigmoid(w.dot(X.T) + b)
    for res in sol:
        if res >= 0.5:
            res = 0
        else:
            res = 1
        idx += 1
        ans.append([idx,res])
    filename = "ansGen.csv"
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

def validDis(Vx,Vy):
    w = np.load('model_w.npy')
    idx = 0
    ans = []
    sqrtsum = 0
    roundsum = 0
    succ = 0
    fail = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i, j in zip(Vx, Vy):
        x = np.array(i, dtype=float)
        res = sigmoid(w.dot(x))
        if res>=0.5:
            res=1
        else:
            res = 0
        if res == j:
            succ+=1
        else:
            fail+=1
        if j==1:
            if res==1:
                TP+=1
            else:
                FN+=1
        else:
            if res==1:
                FP +=1
            else:
                TN+=1
        # sqrtsum += (res - float(j)) ** 2
        # nres = np.round(res)
        # roundsum += (nres - float(j)) ** 2
        # print("id", idx, "result:", round(res,1), "ans:",j, "dist:", res-float(j))
        idx += 1
    print("score:", (succ / idx), "F1-measure:", (2*TP/(2*TP+FP+FN)))
    return (succ / idx)
    # print("round score:", np.sqrt(roundsum / idx))
    # return np.sqrt(sqrtsum / idx)

def Dis(rand=True,split=0,block=0,lamb = 1000,lr = 0.1,bs = 32):
    X, Y, Vx, Vy, T = load_data(0.9, rand, split, block)
    w = np.zeros((X.shape[1],))
    #w = np.load('model_w.npy')

    succ = 0
    fail = 0
    cost = 0
    iteration = 2000
    batch_size = bs
    step = X.shape[0]//batch_size+1
    w_lr = lr
    w_gras = np.ones((X.shape[1],))
    total_cost = 0
    total = 0
    t_start = time.time()
    rem_time = 0
    jseq = []
    tseq = []
    vseq = []
    for j in range(iteration):
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        if j % 50 == 1:
            time_interval = time.time() - t_start
            rem_time = time_interval * (iteration - j) / j
        for idx in range(step):
            x = X[idx*batch_size:(idx+1)*batch_size]
            y = Y[idx*batch_size:(idx+1)*batch_size]
            res = sigmoid(x.dot(w))
            #loss = res - y
            cross_entropy = -(y.dot(np.log(res))+(1-y).dot(np.log(1-res)))

            total_cost += np.sum(cross_entropy)
            total += len(x)
            w_grad = -x.transpose().dot(y - res)
            #print(w_grad.shape)
            w_gras += w_grad**2
            reg = w[:]
            reg[-1] = 0
            reg = reg*lamb/len(reg)
            w = w - w_lr * ((w_grad)/ np.sqrt(w_gras))


            print(
            chr(13) + "|" + "=" * (50 * j // iteration
            ) + ">" + " " * (50 * (iteration - j) // iteration
            ) + "| " + str(
                round(100 * j / iteration, 1)) + "%",
            "\tave cost:",round(total_cost/total,2), "\tremain:",round(rem_time,0),"s",
            sep=' ', end = '', flush = True)
            succ = 0
            fail = 0
            total = 0
            total_cost = 0
        if j%500==1:
            np.save('model_w.npy',w)
            print("\nTrain data ",end="")
            td = validDis(X, Y)
            tseq.append(td)
            print("Valid data ",end="")
            vd = validDis(Vx, Vy)
            vseq.append(vd)
            jseq.append(j)
    print("\n", end="")
    np.save('model_w.npy', w)
    print("Train data ",end="")
    td = validDis(X, Y)
    tseq.append(td)
    print("Valid data ",end="")
    vd = validDis(Vx, Vy)
    vseq.append(vd)
    jseq.append(j)
    return jseq, tseq, vseq

def lr():
    trace = []
    legend = []
    for j in [1,2]:
        for i in range(-3,4):
            trace.append(Dis(lr=10**i))
            legend.append('lr = ' + str(10**i))
        np.save('trace.npy',trace)
        plt.plot(trace[0][0], trace[0][j],
                 trace[1][0], trace[1][j],
                 trace[2][0], trace[2][j],
                 trace[3][0], trace[3][j],
                 trace[4][0], trace[4][j],
                 trace[5][0], trace[5][j],
                 trace[6][0], trace[6][j])
        plt.legend(legend)
        plt.ylabel('ave cost')
        plt.xlabel('epoch')
        if j == 1:
            plt.title('Compare between diff learning rates')
            plt.savefig('w_lr_Training.png')
        if j == 2:
            plt.title('Compare between diff learning rates')
            plt.savefig('w_lr_Validation.png')
        plt.clf()

def load_test():
    test_X = open('test_X')
    X = list(csv.reader(test_X))
    title, X = X[0],X[1:]
    X = np.array(X, dtype=float)
    # X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    np.save('T.npy', X)
    T = np.load('T.npy')
    X = np.load('X.npy')
    X, T = normalize(X, T)
    return T


def testDis():
    w = np.load('model_w.npy')
    _x,_y,_vx,_vy, X = load_data()
    idx = 0
    ans = []

    for i in X:
        x = np.array(i, dtype=float)
        res = sigmoid(w.dot(x))
        if res >= 0.5:
            res = 1
        else:
            res = 0
        idx += 1
        ans.append([idx,res])
    filename = "ans.csv"
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

# def dropout():
