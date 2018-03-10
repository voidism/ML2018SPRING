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
        if mon == 8:
            continue
        for piv in range(471):
            temp = []
            #for term in data:
            #    temp = temp + term[471*mon+piv : 471*mon+piv+9 ]
            temp+=data[5][471*mon+piv+7 : 471*mon+piv+9 ]
            temp+=data[7][471 * mon + piv+7: 471 * mon + piv + 9]
            temp+=data[8][471 * mon + piv: 471 * mon + piv + 9]
            temp+=data[9][471 * mon + piv: 471 * mon + piv + 9]
            for i in data[9][471 * mon + piv: 471 * mon + piv + 9]:
                temp.append(str(float(i)**2))
            temp+=data[10][471 * mon + piv+7: 471 * mon + piv + 9]
            temp.append("1.0")
            collection.append(temp)
            Y.append(data[9][471*mon+piv+9])
    return collection, Y

def load_data():
    datalist = csv_to_matrix("train.csv")
    X, Y = matrix_expansion(datalist)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
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
    X, Y, Vx, Vy = load_data()

    #w = np.zeros((len(X[0]),))
    w = np.load('model_w.npy')

    succ = 0
    fail = 0
    cost = 0
    iteration = 50000
    w_lr = 1000
    w_gras = np.ones((len(X[0]),))
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
        data+=(traindata[18*i+5][-2:])
        data+=(traindata[18*i+7][-2:])
        data+=(traindata[18*i+8][2:])
        data+=(traindata[18*i+9][2:])
        for j in traindata[18*i+9][2:]:
            data.append(str(float(j) ** 2))
        for j in traindata[18*i+10][-2:]:
            if j == "NR":
                data.append(0.0)
            else:
                data.append(j)
        data.append("1.0")
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
        res = np.absolute(res)
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
        res = np.absolute(res)
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
