import csv, sys
import numpy as np
import json
import math
import random
from math import log, floor
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys


global feature_dict
# best -->
feature_dict = [0,#AMB_TEMP
                0,#CH4
                0,#CO
                0,#NMHC
                0,#NO
                2,#NO2
                0,#NOx
                2,#O3
                9,#PM10
                9,#PM2.5
                2,#RAINFALL
                0,#RH
                2,#SO2
                0,#THC
                0,#WD_HR
                0,#WIND_DIREC
                0,#WIND_SPEED
                0,#WS_HR
                9]#PM2.5^2
# feature_dict = [0, 0,0,0,0,0, 0,5,7,9,0,
#                    0,3,0,0,0, 0,0,4]

def csv_to_matrix(filename):
    trainfile = open(filename)
    traindata = csv.reader(trainfile)
    #print(traindata)

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
    global feature_dict
    collection = []
    idx = 0
    Y = []
    f = feature_dict
    for mon in range(12):
        for piv in range(471):
            if '#' in data[9][471 * mon + piv: 471 * mon + piv + 10]:
                continue
            temp = []
            #AMB_TEMP
            temp += data[0][471 * mon + piv + 9 - f[0]: 471 * mon + piv + 9]
            #CH4
            temp += data[1][471 * mon + piv + 9 - f[1]: 471 * mon + piv + 9]
            #CO
            temp += data[2][471 * mon + piv + 9 - f[2]: 471 * mon + piv + 9]
            #NMHC
            temp += data[3][471 * mon + piv + 9 - f[3]: 471 * mon + piv + 9]
            #NO
            temp += data[4][471 * mon + piv + 9 - f[4]: 471 * mon + piv + 9]
            #NO2
            temp += data[5][471 * mon + piv + 9 - f[5]: 471 * mon + piv + 9]
            #NOx
            temp += data[6][471 * mon + piv + 9 - f[6]: 471 * mon + piv + 9]
            #O3
            temp += data[7][471 * mon + piv + 9 - f[7]: 471 * mon + piv + 9]
            #PM10
            temp += data[8][471 * mon + piv + 9 - f[8]: 471 * mon + piv + 9]
            #PM2.5
            temp += data[9][471 * mon + piv + 9 - f[9]: 471 * mon + piv + 9]

            for i in data[9][471 * mon + piv + 9 - f[18]: 471 * mon + piv + 9]:
                temp.append(str(float(i) ** 2))
            #RAINFALL
            temp += data[10][471 * mon + piv + 9 - f[10]: 471 * mon + piv + 9]
            #RH
            temp += data[11][471 * mon + piv + 9 - f[11]: 471 * mon + piv + 9]
            #SO2
            temp += data[12][471 * mon + piv + 9 - f[12]: 471 * mon + piv + 9]
            #THC
            temp += data[13][471 * mon + piv + 9 - f[13]: 471 * mon + piv + 9]
            #WD_HR
            temp += data[14][471 * mon + piv + 9 - f[14]: 471 * mon + piv + 9]
            #WIND_DIREC
            temp += data[15][471 * mon + piv + 9 - f[15]: 471 * mon + piv + 9]
            #WIND_SPEED
            temp += data[16][471 * mon + piv + 9 - f[16]: 471 * mon + piv + 9]
            #WS_HR
            temp += data[17][471 * mon + piv + 9 - f[17]: 471 * mon + piv + 9]
            temp.append("1.0")
            # temp+=data[5][471*mon+piv+9-2 : 471*mon+piv+9 ]
            # temp+=data[7][471 * mon + piv+9-2: 471 * mon + piv + 9]
            # temp+=data[8][471 * mon + piv+9-1: 471 * mon + piv + 9]
            # for i in data[8][471 * mon + piv+9-1: 471 * mon + piv + 9]:
            #     temp.append(str(float(i)**2))
            # temp+=data[9][471 * mon + piv: 471 * mon + piv + 9]
            # for i in data[9][471 * mon + piv+9-3: 471 * mon + piv + 9]:
            #     temp.append(str(float(i)**2))
            # temp+=data[10][471 * mon + piv+9-1: 471 * mon + piv + 9]
            # temp.append("1.0")
            collection.append(temp)
            Y.append(data[9][471*mon+piv+9])
    return collection, Y

def load_data(rand=True,split=0,block=0,whole=False):
    datalist = csv_to_matrix("train.csv")
    X, Y = matrix_expansion(datalist)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    v_size = 0
    if rand:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X,Y = (X[randomize], Y[randomize])

    if whole:
        v_size = floor(len(X)*0.95)
        Vx = X[v_size:]
        Vy = Y[v_size:]

    elif split==0:
        v_size = floor(len(X)*0.95)
        Vx = X[v_size:]
        Vy = Y[v_size:]
        X = X[:v_size]
        Y = Y[:v_size]
    elif split!=0:
        v_size = floor(len(X) // split)
        block-=1
        if block not in range(split):
            block = 0
            print("FUCK please choose the right block!")
        Vx = X[block*v_size:(block+1)*v_size]
        Vy = Y[block*v_size:(block+1)*v_size]
        X = np.concatenate((X[:block*v_size],X[(block+1)*v_size:]))
        Y = np.concatenate((Y[:block*v_size],Y[(block+1)*v_size:]))

    return X,Y,Vx,Vy



def load_pm25_only(rand=True,split=0,block=0,whole=False):
    data = csv_to_matrix("train.csv")
    X = []
    idx = 0
    Y = []
    for mon in range(12):
        for piv in range(471):
            if '#' in data[9][471 * mon + piv: 471 * mon + piv + 10]:
                continue
            temp = []
            temp += data[9][471 * mon + piv: 471 * mon + piv + 9]
            for i in data[9][471 * mon + piv: 471 * mon + piv + 9]:
                temp.append(str(float(i) ** 2))
            temp.append("1.0")
            X.append(temp)
            Y.append(data[9][471 * mon + piv + 9])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    v_size = 0
    if rand:
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X, Y = (X[randomize], Y[randomize])

    if whole:
        v_size = floor(len(X) * 0.95)
        Vx = X[v_size:]
        Vy = Y[v_size:]

    elif split == 0:
        v_size = floor(len(X) * 0.95)
        Vx = X[v_size:]
        Vy = Y[v_size:]
        X = X[:v_size]
        Y = Y[:v_size]
    elif split != 0:
        v_size = floor(len(X) // split)
        block -= 1
        if block not in range(split):
            block = 0
            print("FUCK please choose the right block!")
        Vx = X[block * v_size:(block + 1) * v_size]
        Vy = Y[block * v_size:(block + 1) * v_size]
        X = np.concatenate((X[:block * v_size], X[(block + 1) * v_size:]))
        Y = np.concatenate((Y[:block * v_size], Y[(block + 1) * v_size:]))
    return X, Y, Vx, Vy


def best(rand=True,split=0,block=0,whole=False,pm25=False):
    x, y, Vx, Vy = load_data(rand, split, block, whole)
    if pm25:
        x, y, Vx, Vy = load_pm25_only(rand, split, block, whole)

    w = np.zeros(len(x[0]))
    w = np.matmul(np.matmul(inv(np.matmul(x.T, x) + 1078 * np.identity(len(x[0]))), x.T), y)
    np.save('model_w.npy', w)
    test(pm25)

def train(rand=True,split=0,block=0,whole=False,pm25=False,lamb = 1000):
    X, Y, Vx, Vy = load_data(rand, split, block, whole)
    if pm25:
        X, Y, Vx, Vy = load_pm25_only(rand,split,block,whole)

    w = np.zeros((len(X[0]),))
    #w = np.load('model_w.npy')

    succ = 0
    fail = 0
    cost = 0
    iteration = 100000
    w_lr = 700
    w_gras = np.ones((len(X[0]),))
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
        reg = w[:]
        reg[-1] = 0
        reg = reg*lamb/len(reg)
        w = w - w_lr * ((w_grad)/ np.sqrt(w_gras))

        print(
        chr(13) + "|" + "=" * (50 * j // iteration
        ) + ">" + " " * (50 * (iteration - j) // iteration
        ) + "| " + str(
            round(100 * j / iteration, 1)) + "%",
        "\tave cost:",total_cost/total,
        sep=' ', end = '', flush = True)
        succ = 0
        fail = 0
        total = 0
        total_cost = 0
    print("\n", end="")
    np.save('model_w.npy',w)
    print("===valid===")
    return valid(Vx,Vy)



def seek_zero(filename):
    f = open(filename)
    r =csv.reader(f)
    l = list(r)
    n = []
    for i in l:
        if i[1] == 'RAINFALL':
            continue
        else:
            for j in i[2:]:
                if float(j)<=0:
                    n.append(i)
                    break
    ff = open('zero_data.csv','w')
    s = csv.writer(ff, delimiter=',', lineterminator='\n')
    for k in n:
        s.writerow(k)
    ff.close()


def preprocess(filename):
    f = open(filename)
    r =csv.reader(f)
    l = list(r)
    new = []
    for i in l:
        if i[1] == 'RAINFALL':
            temprow = []
            for j in i[2:]:
                if j == "NR":
                    temprow.append(0.0)
                else:
                    temprow.append(float(j))
            new.append(np.array(temprow))
        else:
            temprow = np.array(i[2:],dtype = float)
            tf = temprow<=0
            if True in tf:
                if False not in tf:
                    pass
                    #print(l.index(i),temprow)
                else:
                    #print(l.index(i), temprow)
                    if temprow[-2]==11 and temprow[-3]==8 and temprow[-4] == 5:
                        temprow[-1] = 14
                    elif temprow[-2]==24 and temprow[-3]==31 and temprow[-4] == 40:
                        temprow[-1] = 22
                    elif temprow[-1]==0.3 and temprow[-2]==0 and temprow[-3] == 11:
                        pass
                    elif temprow[0]==45 and temprow[1]==77 and temprow[2] == 83:
                        temprow[-2]=78
                        temprow[-3]=95
                    else:
                        temprow = np.where(temprow<-1, -temprow, temprow)
                        temprow = np.where(((temprow>=-1)&(temprow<=0)),np.nan,temprow)
                        s = pd.Series(temprow)
                        s = s.interpolate()
                        temprow = np.array(s)

                        if True in np.isnan(temprow):
                            idx = np.argmax(np.isnan(temprow)==False)
                            temprow = np.where(np.isnan(temprow)==True, temprow[idx], temprow)
                        #print(l.index(i), temprow)
            new.append(temprow)

    topic = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10',
             'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC',
             'WIND_SPEED', 'WS_HR']
    filename = "pre_test.csv"
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    for i in range(len(new)):
        dummy = np.where(new[i] == np.round(new[i]), new[i].astype(int).astype(str), new[i].astype(str))
        dummy = np.where(dummy == '0', 'NR', dummy)
        s.writerow(["id_"+str(i//18),topic[i%18]]+list(dummy))
    text.close()
    return new

def see():
    f1 = open('test.csv')
    f2 = open('pre_test.csv')
    r1 = csv.reader(f1)
    r2 = csv.reader(f2)
    l1 = list(r1)
    l2 = list(r2)
    for i, j in zip(l1, l2):
        if i != j:
            print(i, j)


def test_to_matrix(filename,pm25=False):
    global feature_dict
    f = feature_dict
    traindata = preprocess(filename)
    topdata = []
    idx = 0
    if not pm25:
        for i in range(260):
            data = []
            data.append(traindata[18 * i + 0][9-f[0]:])
            data.append(traindata[18 * i + 1][9-f[1]:])
            data.append(traindata[18 * i + 2][9-f[2]:])
            data.append(traindata[18 * i + 3][9-f[3]:])
            data.append(traindata[18 * i + 4][9-f[4]:])
            data.append(traindata[18 * i + 5][9-f[5]:])
            data.append(traindata[18 * i + 6][9-f[6]:])
            data.append(traindata[18 * i + 7][9-f[7]:])
            data.append(traindata[18 * i + 8][9-f[8]:])
            data.append(traindata[18 * i + 9][9-f[9]:])

            #for j in traindata[18*i+9][9-f[9]:]:
                #data.append(str(float(j) ** 2))
            data.append((traindata[18 * i + 9][9 - f[18]:])**2)

            data.append(traindata[18 * i + 10][9-f[10]:])

            data.append(traindata[18 * i + 11][9-f[11]:])
            data.append(traindata[18 * i + 12][9-f[12]:])
            data.append(traindata[18 * i + 13][9-f[13]:])
            data.append(traindata[18 * i + 14][9-f[14]:])
            data.append(traindata[18 * i + 15][9-f[15]:])
            data.append(traindata[18 * i + 16][9-f[16]:])
            data.append(traindata[18 * i + 17][9-f[17]:])
            # data+=(traindata[18*i+5][-2:])
            # data+=(traindata[18*i+7][-2:])
            # data+=(traindata[18*i+8][-1:])
            # for j in traindata[18*i+8][-1:]:
            #     data.append(str(float(j) ** 2))
            # data+=(traindata[18*i+9][-9:])
            # for j in traindata[18*i+9][-3:]:
            #     data.append(str(float(j) ** 2))
            data.append(np.array([1.0]))
            data = np.concatenate(data)
            topdata.append(data)
            idx+=1
    elif pm25:
        for i in range(260):
            data = []
            data += (traindata[18 * i + 9][2:])
            for j in traindata[18 * i + 9][2:]:
                data.append(str(float(j) ** 2))
            data.append("1.0")
            topdata.append(data)
            idx += 1
    return topdata

def valid(Vx,Vy):
    w = np.load('model_w.npy')
    idx = 0
    ans = []
    sqrtsum = 0
    roundsum = 0
    for i,j in zip(Vx, Vy):
        x = np.array(i, dtype=float)
        res = w.dot(x)
        #res = np.absolute(res)
        sqrtsum += (res-float(j))**2
        nres = np.round(res)
        roundsum += (nres - float(j)) ** 2
        #print("id", idx, "result:", round(res,1), "ans:",j, "dist:", res-float(j))
        idx += 1
    print("score:",np.sqrt(sqrtsum/idx))
    #print("round score:", np.sqrt(roundsum / idx))
    return np.sqrt(sqrtsum/idx)

def trick():
    trainfile = open('test.csv')
    traindata = list(csv.reader(trainfile))
    topdata = []
    for i in range(260):
        topdata.append(traindata[18 * i + 9][-2:])
    idx = 0
    ans = []
    tests = topdata
    for i in tests:
        x = np.array(i,dtype=float)
        res = (2*x[1]-x[0])
        print("id",idx,"x[0]:",x[0],"x[1]:",x[1],"result:",res)
        ans.append(["id_"+str(idx),res])
        idx+=1

    filename = "ans_trick.csv"
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "value"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

def test(pm25=False):
    w = np.load('model_w.npy')
    tests = test_to_matrix("raw_test.csv",pm25)
    #print(tests)
    idx = 0
    ans = []
    for i in tests:
        x = np.array(i,dtype=float)
        res = w.dot(x)
        #res = np.absolute(res)
        if res < 0:
            res = 0.0
        nres = np.round(res)
        print("id",idx,"result:",res, "round:", nres)
        ans.append(["id_"+str(idx),res])
        idx+=1

    filename = "ans.csv"
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "value"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()


if __name__ == '__main__':
    if sys.argv[3]=="--best":
        feature_dict = [0, 0, 0, 0, 0, 2, 0, 2, 9, 9, 2,
                        0, 2, 0, 0, 0, 0, 0, 9]
        w = np.load('model_w_HW1BEST.npy')
    else:
        feature_dict = [0, 0, 0, 0, 0, 2, 0, 2, 9, 9, 2,
                        0, 2, 0, 0, 0, 0, 0, 9]
        w = np.load('model_w_HW1.npy')
    tests = test_to_matrix(sys.argv[1])
    idx = 0
    ans = []
    for i in tests:
        x = np.array(i,dtype=float)
        res = w.dot(x)
        #res = np.absolute(res)
        if res < 0:
            res = 0.0
        nres = np.round(res)
        #print("id",idx,"result:",res, "round:", nres)
        ans.append(["id_"+str(idx),res])
        idx+=1

    filename = sys.argv[2]
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "value"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()