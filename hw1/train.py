import csv
import numpy as np
import json
import math
import random
from math import log, floor


def csv_to_matrix(filename):
    trainfile = open(filename)
    traindata = csv.reader(trainfile)
    print(traindata)

    data = []
    idx = 0
    for i in traindata:
        if idx == 0:
            pass
        elif idx<19:
            temp = []
            for j in i[3:]:
                temp.append(0.0 if j=="NR" else j)
            data.append(temp)
        else:
            for j in i[3:]:
                data[(idx-1)%18].append(0.0 if j=="NR" else j)
        idx+=1

    return data

def matrix_expansion(data):
    collection = []
    idx = 0
    Y = []
    for mon in range(12):
        for piv in range(471):
            temp = []
            for term in data:
                temp = temp + term[471*mon+piv : 471*mon+piv+9 ]
            temp.append("1.0")
            collection.append(temp)
            Y.append(data[9][471*mon+piv+9])
    return collection, Y

def readin_X_Y():
    datalist = csv_to_matrix("train.csv")
    X, Y = matrix_expansion(datalist)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X,Y = (X[randomize], Y[randomize])
    v_size = floor(len(X)*0.95)
    Vx = X[v_size:]
    Vy = Y[v_size:]
    X = X[:v_size]
    Y = Y[:v_size]
    return X,Y,Vx,Vy

def train():
    #datalist = csv_to_matrix("train.csv")
    #print(datalist)
    X, Y, Vx, Vy = readin_X_Y()

    #w = np.full((18,9),0)
    w = np.zeros((18*9+1,))
    #b = 0.0
    #print(w)

    succ = 0
    fail = 0
    cost = 0
    iteration = 1000
    w_lr = 10
    w_gras = np.ones((18*9+1,))
    #b_gras = 1
    total_cost = 0
    total = 0
    for j in range(iteration):
        x = np.array(X, dtype=float)
        y = np.array(Y, dtype=float)
        res = x.dot(w)
        loss = res - y
        cost = math.sqrt(np.sum(loss ** 2) / len(x))
        total_cost += cost
        total += 1
        w_grad = x.transpose().dot(loss)
        w_gras += w_grad**2
        w = w - w_lr * (w_grad / np.sqrt(w_gras))
        '''
        for xo,y in zip(X,Y):
            x = np.array(xo,dtype=float)
            res = w.dot(x)
            #res = resmat.sum() + b
            loss = (float(y) - res)
            if loss==float('Inf'):
                print("=========Fail=========")
                break
            cost = math.sqrt(loss**2/(18*9+1))
            total_cost += cost
            total+=1
            if cost > 1:
                fail +=1
            else:
                succ +=1
            w_grad = -2*loss*x
            w_gras += w_grad**2
            w = w - w_lr*( w_grad / np.sqrt(w_gras) )
        '''
        try:
            print("cost:", cost,"\tave cost:",total_cost/total)
            succ = 0
            fail = 0
            total = 0
            total_cost = 0
        except:
            pass

    np.save('model_w.npy',w)


def test_to_matrix(filename):
    trainfile = open(filename)
    traindata = list(csv.reader(trainfile))
    topdata = []
    idx = 0
    for i in range(260):
        data = []
        for j in range(18):
            for w in traindata[18*i+j][2:]:
                if w == "NR":
                    data.append(0.0)
                else:
                    data.append(w)
        data.append(1.0)
        topdata.append(data)
        idx+=1

    return topdata

def valid(Vx,Vy):
    w = np.load('model_w.npy')
    idx = 0
    ans = []
    sqrtsum = 0
    for i,j in zip(Vx, Vy):
        x = np.array(i, dtype=float)
        res = w.dot(x)
        sqrtsum += (res-float(j))**2
        print("id", idx, "result:", res, "ans:",j)
        idx += 1
    print("score:",np.sqrt(sqrtsum/idx))

def test():
    w = np.load('model_w.npy')
    tests = test_to_matrix("test.csv")
    idx = 0
    ans = []
    for i in tests:
        x = np.array(i,dtype=float)
        res = w.dot(x)
        print("id",idx,"result:",res)
        if res<0:
            res = 0
        elif res>100:
            res = 100
        ans.append(["id_"+str(idx),res])
        idx+=1

    filename = "ans.csv"
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "value"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()
