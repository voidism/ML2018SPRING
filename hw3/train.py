import numpy as np
import csv
import math
import random
from math import log, floor
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
# from keras_vggface import utils
from keras.engine.topology import get_source_inputs
import warnings
from keras.models import Model
from keras import layers
from keras.models import load_model


def load_data(train_data_path='train.csv',test_data_path='test.csv'):
    r = csv.reader(open(train_data_path))
    l = list(r)[1:]
    X_train = []
    Y_pre = []
    for row in l:
        Y_pre.append(row[0])
        X_train.append(np.reshape(np.array(row[1].split(' '),dtype=float),(48,48,1)))
    X_train = np.array(X_train,dtype=float)

    # one-hot encoding
    Y_pre = np.array(Y_pre,dtype=int)
    Y_train = np.zeros((Y_pre.shape[0],7))
    Y_train[np.arange(Y_pre.shape[0]), Y_pre] = 1

    r = csv.reader(open(test_data_path))
    l = list(r)[1:]
    X_test = []
    for row in l:
        X_test.append(np.reshape(np.array(row[1].split(' '),dtype=float),(48,48,1)))
    X_test = np.array(X_test,dtype=float)

    X_train /= 255
    X_test /= 255

    return X_train, Y_train, X_test

def split_valid(X,Y,v_size=0.9,rand=False,split=0,block=0):
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
            print("please choose the right block!")
        Vx = X[block*v_size:(block+1)*v_size]
        Vy = Y[block*v_size:(block+1)*v_size]
        X = np.concatenate((X[:block*v_size],X[(block+1)*v_size:]))
        Y = np.concatenate((Y[:block*v_size],Y[(block+1)*v_size:]))
    
    return X, Y, Vx, Vy


def build_model(x_train):
    num_classes = 7

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    return model

def train_model(model,x_train,y_train,x_test,y_test):
    # initiate RMSprop optimizer
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=500,
              epochs=100,
              validation_data=(x_test, y_test),
              shuffle=True)
    return model

def test(filename = "ans.csv"):
    ans = []
    _x, _y, test = load_data()
    model = load_model('my_model.h5')
    result = np.argmax(model.predict(test),axis=1)
    for idx in range(result.shape[0]):
        ans.append([idx,result[idx]])

    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()
    
if __name__=="__main__":
    X_org, Y_org, X_test = load_data()
    X_train, Y_train, X_valid, Y_valid = split_valid(X_org, Y_org)
    if sys.argv[1]=='-init':
        model = build_model(X_train)
    elif sys.argv[1]=='-cont':
        model = load_model('my_model.h5')
    else:
        print('No Arguments!')
        sys.exit()
    model = train_model(model, X_train, Y_train, X_valid, Y_valid)
    model.save('my_model.h5')
