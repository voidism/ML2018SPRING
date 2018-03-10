from math import floor
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, SGD
import math
import csv
from keras.models import load_model

def load_data(rand=False):
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    X = X[:,:-1]
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    Y = Y.clip(max=120,min=0)
    X = (X - X.min(0)) / X.ptp(0)
    # for vec in X.T:
    #     mins = np.min(vec, axis=0)
    #     maxs = np.max(vec, axis=0)
    #     rng = maxs - mins
    #     vec = (vec - mins)/(maxs - mins)
    if(rand):
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X,Y = (X[randomize], Y[randomize])
    v_size = floor(len(X)*0.95)
    Vx = X[v_size:]
    Vy = Y[v_size:]
    X = X[:v_size]
    Y = Y[:v_size]
    return X,Y,Vx,Vy

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

def load_test():
    tests = test_to_matrix("test.csv")
    tests = np.array(tests, dtype=float)
    tests = tests[:, :-1]
    tests = (tests - tests.min(0)) / tests.ptp(0)
    return tests

def test():
    idx = 0
    ans = []
    tests = load_test()
    model = load_model('my_model.h5')
    result = np.squeeze(model.predict(tests))
    for i in result:
        ans.append(["id_"+str(idx),i])
        idx+=1

    filename = "NNans.csv"
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "value"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

def valid():
    for i in range(10):
        print("====Val", i, "====")
        x_train,y_train,x_test,y_test=load_data(rand = True)
        model = load_model('my_model.h5')
        result = np.squeeze(model.predict(x_test))
        err = 0
        for i, j in zip(result, y_test):
            err += (i - j) ** 2
            print(i, j)
        print("err", err)
        print("rmse", math.sqrt(err / len(y_test)))

def trainval():
    x_train,y_train,x_test,y_test=load_data()
    model = load_model('my_model.h5')
    result = np.squeeze(model.predict(x_train))
    err = 0
    for i, j in zip(result, y_train):
        err += (i - j) ** 2
        print(i, j)
    print("err", err)
    print("rmse", math.sqrt(err / len(y_train)))


def build_model():
        #建立模型
        model = Sequential()
        #將模型疊起
        model.add(Dense(input_dim=33,units=50, use_bias=True,activation='relu'))
        model.add(Dense(units=40, use_bias=True,activation='relu'))
        model.add(Dense(units=30, use_bias=True, activation='relu'))
        model.add(Dense(units=20, use_bias=True, activation='relu'))
        model.add(Dense(units=10, use_bias=True,activation='relu'))
        model.add(Dense(units=1, use_bias=True))
        model.summary()
        return model

def trainNN(cont = True):
    x_train,y_train,x_test,y_test=load_data(rand=True)
    if cont:
        model = load_model('my_model.h5')
    else:
        model = build_model()
    #開始訓練模型
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=50,epochs=600)
    model.save('my_model.h5')
    #顯示訓練結果
    score = model.evaluate(x_train,y_train)
    print ('\nTrain Acc:', score[1])
    score = model.evaluate(x_test,y_test)
    print ('\nTest Acc:', score[1])
    result = np.squeeze(model.predict(x_test))
    print(result)
    err = 0
    for i,j in zip(result,y_test):
        err += (i - j)**2
        print(i,j)
    print("err",err)
    print("rmse",math.sqrt(err / len(y_test)))
    test()