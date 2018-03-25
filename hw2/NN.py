from math import floor
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, SGD, RMSprop
import math
import csv
from keras.models import load_model
from keras import regularizers

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    X_train_test = np.delete(X_train_test,116,1)
    X_train_test = np.delete(X_train_test,54,1)
    X_train_test = np.delete(X_train_test,np.s_[27:43],1)
    X_train_test = np.delete(X_train_test,7,1)
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

def test():
    idx = 0
    ans = []
    _x,_y,_vx,_vy,tests = load_data()
    model = load_model('my_model.h5')
    result = np.squeeze(model.predict(tests))
    for i in result:
        ans.append([idx+1,1 if i>0.5 else 0])
        idx+=1

    filename = "NNans.csv"
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()

def valid():
    for i in range(10):
        print("====Val", i, "====")
        x_train,y_train,x_test,y_test,_T=load_data(rand = True)
        model = load_model('my_model.h5')
        result = np.squeeze(model.predict(x_test))
        err = 0
        for i, j in zip(result, y_test):
            if i>0.5:
                i = 1
            else:
                i = 0
            err += (i - j) ** 2
        print("err", err)
        print("mse", err / len(y_test))

def trainval():
    x_train,y_train,x_test,y_test,_T=load_data()
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
        model.add(Dense(input_dim=105,units=12, activation='sigmoid'))
        model.add(Dense(input_dim=12,units=1, activation='sigmoid'))
        model.summary()
        return model

def trainNN(cont = True):
    x_train,y_train,x_test,y_test,_T=load_data(0.9)
    if cont:
        model = load_model('my_model.h5')
    else:
        model = build_model()
    #開始訓練模型
    model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=50,epochs=40)
    model.save('my_model.h5')
    #顯示訓練結果
    score = model.evaluate(x_train,y_train)
    print ('\nTrain Acc:', score)
    score = model.evaluate(x_test,y_test)
    print ('\nTest Acc:', score)
    result = np.squeeze(model.predict(x_test))
    # print(result)
    err = 0
    for i, j in zip(result, y_test):
        if i>0.5:
            i = 1
        else:
            i = 0
        err += (i - j) ** 2
    print("err", err)
    print("mse", err / len(y_test))

trainNN(0)
test()