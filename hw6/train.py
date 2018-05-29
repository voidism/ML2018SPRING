import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate, Dropout, Dense
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import csv
from math import floor
from output import output
from keras import backend as K
import keras.objectives

global _mean, _std

# def RMSE(y_true, y_pred):
#     global _mean, _std
#     y_true = y_true*_std + _mean
#     y_pred = y_pred*_std + _mean
#     y_pred = K.clip(y_pred, 1., 5.)
#     return K.sqrt(K.mean(K.square((y_pred - y_true)*_std), axis=-1)) 


# keras.objectives.custom_loss = rmse

def build_model(user_dim, movie_dim, latent_dim,sgd):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    user_embed = Embedding(user_dim, latent_dim, embeddings_regularizer=l2(0.00001))(user_input)
    user_vec = Flatten()(user_embed)
    user_vec = Dropout(0.5)(user_vec)

    movie_embed = Embedding(movie_dim, latent_dim, embeddings_regularizer=l2(0.00001))(movie_input)
    movie_vec = Flatten()(movie_embed)
    user_vec = Dropout(0.5)(user_vec)
    user_bias = Flatten()(Embedding(user_dim, 1)(user_input))
    movie_bias = Flatten()(Embedding(movie_dim, 1)(movie_input))
    res = Dot(axes=1)([user_vec, movie_vec])
    res = Add()([res, user_bias, movie_bias])
    # res = Dense(1,activation='linear')
    model = Model([user_input, movie_input], res)
    def rmse(y_true, y_pred): return K.sqrt( K.mean(((y_pred - y_true)*_std)**2) )
    model.compile(loss='mse', optimizer=sgd, metrics=[rmse])#, embeddings_initializer='random_normal'
    return model

def build_nn_model(user_dim, movie_dim, latent_dim,sgd):
    user_input = Input(shape=[1])
    movie_input = Input(shape=[1])
    user_embed = Embedding(user_dim, latent_dim, embeddings_regularizer=l2(0.00001))(user_input)
    user_vec = Flatten()(user_embed)

    movie_embed = Embedding(movie_dim, latent_dim, embeddings_regularizer=l2(0.00001))(movie_input)
    movie_vec = Flatten()(movie_embed)
    # user_bias = Flatten()(Embedding(user_dim, 1, embeddings_initializer='zeros')(user_input))
    # movie_bias = Flatten()(Embedding(movie_dim, 1, embeddings_initializer='zeros')(movie_input))
    
    res = Concatenate()([user_vec, movie_vec])
    res = Dense(128, activation='relu')(res)
    res = Dropout(0.33)(res)
    res = Dense(64,activation='relu')(res)
    res = Dropout(0.33)(res)
    res = Dense(32,activation='relu')(res)
    res = Dropout(0.33)(res)
    res = Dense(1, activation='linear')(res)

    # res = Add()([res, user_bias, movie_bias])
    model = Model([user_input, movie_input], res)
    def rmse(y_true, y_pred): return K.sqrt( K.mean(((y_pred - y_true)*_std)**2) )
    model.compile(loss='mse', optimizer=sgd, metrics=[rmse])
    return model

def split_valid(X,Y,Z,v_size=0.95,rand=True):
    if rand:
        np.random.seed(9487)
        # randomize = np.arange(len(X))
        # np.random.shuffle(randomize)
        randomize = np.random.permutation(len(X))
        X,Y,Z = (X[randomize], Y[randomize], Z[randomize])
    
    v_size = floor(len(X) * v_size)
    Vx = X[v_size:] if v_size != len(X) else X[floor(len(X) * 0.9):]
    Vy = Y[v_size:] if v_size != len(X) else Y[floor(len(X) * 0.9):]
    Vz = Z[v_size:] if v_size != len(X) else Z[floor(len(X) * 0.9):]
    X = X[:v_size]
    Y = Y[:v_size]
    Z = Z[:v_size]
    
    return X, Y, Z, Vx, Vy, Vz


def load_data():
    r = csv.reader(open('train.csv'))
    l = list(r)[1:]
    a = np.array(l, dtype=float)
    rate = np.array(a.T[3][:])
    rate = (rate - rate.mean()) / rate.std()
    return a.T[1] - 1, a.T[2] - 1, rate

def mean_std():
    r = csv.reader(open('train.csv'))
    l = list(r)[1:]
    a = np.array(l, dtype=float)
    rate = np.array(a.T[3][:])
    return rate.mean(), rate.std()

_mean, _std = mean_std()

def load_test(name = 'test.csv'):
    r = csv.reader(open(name))
    l = list(r)[1:]
    a = np.array(l, dtype=float)
    return a.T[1] - 1, a.T[2] - 1
    
def test(in_name='test.csv', out_name='ans.csv', model_name='model.h5'):
    def rmse(y_true, y_pred): return K.sqrt( K.mean(((y_pred - y_true)*_std)**2) )
    #model.compile(loss='mse', optimizer=sgd, metrics=[rmse])
    model = load_model(model_name, custom_objects={'rmse':rmse})
    x, y = load_test()
    res = model.predict([x, y], batch_size=10000)

    mean, std = mean_std()
    res = res * std + mean
    res = res.flatten()
    ans = []
    idx = 1
    for i in range(len(res)):
        if res[i]<= 1:
            res[i] = 0.0
        elif res[i]>5:
            res[i] = 5.0
        ans.append([idx,res[i]])
        idx+=1
    output(ans,header=['TestDataID','Rating'],name=out_name)

    
def retrive_extra(user='users.csv', movie='movies.csv'):
    r = open(movie, encoding='latin-1').read()
    l = r.split('\n')[1:-1]
    m_data = []
    for i in range(len(l)):
        m_data.append(l[i].split('::')[2].split('|'))
    return m_data

def train(mf=False,nn=False,extra=False,bs=10000,dim=16,epo=30,sgd='adam',whole=False):
    x, y, z = load_data()
    vx = None
    vy = None
    vz = None
    if not whole:
        x, y, z, vx, vy, vz = split_valid(x, y, z, rand= True)
    model = None
    model_name = 'model.h5'
    if mf:
        model_name = 'mf_model.h5'
        # sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model = build_model(int(np.max(x) + 1), int(np.max(y) + 1), dim, sgd)
    elif nn:
        model_name = 'nn_model.h5'
        model = build_nn_model(int(np.max(x) + 1), int(np.max(y) + 1), dim, sgd)
    cb = [EarlyStopping(monitor='val_rmse', patience=30, verbose=1, mode='min'),
    ModelCheckpoint(filepath=model_name, monitor='val_rmse', mode='min',save_best_only=True)]
    if not whole:
        model.fit([x, y],z, batch_size=bs, epochs=epo, validation_data=([vx,vy],vz), callbacks=cb, verbose=1)
    else:
        model.fit([x, y],z, batch_size=bs, epochs=epo, callbacks=cb, verbose=1)

    model.save('model.h5')

if __name__ == '__main__':
    train(nn=True)
    test()
    
    
