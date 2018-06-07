import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate, Dropout, Dense
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import csv
from math import floor
from output import output
from keras import backend as K
import keras.objectives
import pandas as pd

global _mean, _std

# def RMSE(y_true, y_pred):
#     global _mean, _std
#     y_true = y_true*_std + _mean
#     y_pred = y_pred*_std + _mean
#     y_pred = K.clip(y_pred, 1., 5.)
#     return K.sqrt(K.mean(K.square((y_pred - y_true)*_std), axis=-1)) 


# keras.objectives.custom_loss = rmse

def build_model(user_dim, movie_dim, latent_dim,sgd,bias=True):
    user_input = Input(shape=[1])
    movie_input = Input(shape=[1])
    user_embed = Embedding(user_dim, latent_dim, embeddings_regularizer=l2(0.00001))(user_input)
    user_vec = Flatten()(user_embed)
    user_vec = Dropout(0.5)(user_vec)

    movie_embed = Embedding(movie_dim, latent_dim, embeddings_regularizer=l2(0.00001))(movie_input)
    movie_vec = Flatten()(movie_embed)
    user_vec = Dropout(0.5)(user_vec)
    user_bias = Flatten()(Embedding(user_dim, 1)(user_input))
    movie_bias = Flatten()(Embedding(movie_dim, 1)(movie_input))
    res = Dot(axes=1)([user_vec, movie_vec])
    if bias:
        res = Add()([res, user_bias, movie_bias])
    # res = Dense(1,activation='linear')
    model = Model([user_input, movie_input], res)
    def rmse(y_true, y_pred): return K.sqrt( K.mean(((y_pred - y_true)*_std)**2) )
    model.compile(loss='mse', optimizer=sgd, metrics=[rmse])#, embeddings_initializer='random_normal'
    return model

def build_ex_model(user_dim, movie_dim, latent_dim,sgd):
    user_mat, movie_mat = retrive_extra()

    user_input = Input(shape=[1])
    movie_input = Input(shape=[1])
    user_embed = Embedding(user_dim, latent_dim, embeddings_regularizer=l2(0.00001))(user_input)
    
    user_info = Embedding(user_dim,user_mat.shape[1],weights=[user_mat],trainable=False)(user_input)
    user_info_vec = Dense(latent_dim,activation = 'linear')(user_info)
    
    user_vec = Flatten()(Add()([user_embed, user_info_vec]))
    user_vec = Dropout(0.5)(user_vec)

    movie_embed = Embedding(movie_dim, latent_dim, embeddings_regularizer=l2(0.00001))(movie_input)
    
    movie_info = Embedding(movie_dim,movie_mat.shape[1],weights=[movie_mat],trainable=False)(movie_input)
    movie_info_vec = Dense(latent_dim,activation = 'linear')(movie_info)
    
    movie_vec = Flatten()(Add()([movie_embed,movie_info_vec]))
    movie_vec = Dropout(0.5)(movie_vec)
    user_bias = Flatten()(Embedding(user_dim, 1)(user_input))
    movie_bias = Flatten()(Embedding(movie_dim, 1)(movie_input))
    res = Dot(axes=1)([user_vec, movie_vec])
    res = Add()([res, user_bias, movie_bias])
    # res = Dense(1,activation='linear')
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


def load_data(nor=True):
    r = csv.reader(open('train.csv'))
    l = list(r)[1:]
    a = np.array(l, dtype=float)
    rate = np.array(a.T[3][:])
    if nor:
        rate = (rate - rate.mean()) / rate.std()
    return a.T[1] - 1, a.T[2] - 1, rate

def mean_std(name=''):
    if name=='':
        m,s = np.load('mean_std.npy')
        return m, s
    else:
        r = csv.reader(open(name))
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
    
def test(in_name='test.csv', out_name='ans.csv', model_name='model.h5',nor=True):
    def rmse(y_true, y_pred): return K.sqrt( K.mean(((y_pred - y_true)*_std)**2) )
    #model.compile(loss='mse', optimizer=sgd, metrics=[rmse])
    model = load_model(model_name, custom_objects={'rmse':rmse})
    x, y = load_test()
    res = model.predict([x, y], batch_size=10000)
    if nor:
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
    index = []
    yr = []
    for i in range(len(l)):
        line = l[i].split('::')
        index.append(int(line[0]))
        m_data.append(line[2].split('|'))
        yr.append(int(line[1][-5:-1]))

    yr = np.array(yr, dtype=float)
    yr = (yr - yr.mean()) / yr.std()


    plane = [j for i in m_data for j in i]
    category = []
    for i in plane:
        if i not in category:
                category.append(i)
    index = np.array(index, dtype=int)
    movie_mat = np.zeros((index.max(),len(category)+1))
    for i in range(len(index)):
        one = [category.index(j) for j in m_data[i]]
        movie_mat[index[i]-1, one] = 1
    for i in range(len(movie_mat)):
        if movie_mat[i].sum():
            movie_mat[i]/=movie_mat[i].sum()
    for i in range(len(index)):
        movie_mat[index[i]-1, -1] = yr[i] 

    # users.csv
    u_data = pd.read_csv('users.csv',sep='::',engine='python').values
    index = np.array(u_data.T[0],dtype=int)
    occup = np.array(u_data.T[3],dtype=int)
    age = np.array(u_data.T[2], dtype=float)
    age = (age - age.mean()) / age.std()

    user_mat = np.zeros((index.max(),33))
    for i in range(len(index)):
        user_mat[index[i]-1,0] = float(u_data.T[1][i]=='M')
        user_mat[index[i]-1,1] = age[i]
        user_mat[index[i]-1,2+occup[i]] = 1
        user_mat[index[i]-1,23+int(u_data.T[4][i][0])] = 1

    return user_mat, movie_mat

def exam_dim(rg=[16,32,64,128,256,512,1024]):
    total_sc = []
    for i in rg:
        his = train(mf=True,dim=i)
        nhis = np.array(his.history['val_rmse'])
        score = np.round(np.min(nhis),5)
        idx = np.argmin(nhis)
        total_sc.append([i,idx,score])
        test(model_name='mf_model.h5', out_name='./to_sub/ans_dim_%d_val_%f_epo_%d.csv'%(i,score,idx))
    return total_sc


def train(mf=False,ex=False,extra=False,bs=10000,dim=16,epo=300,sgd='adam',whole=False, nor=True, bias=True):
    x, y, z = load_data(nor=nor)
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
        model = build_model(int(np.max(x) + 1), int(np.max(y) + 1), dim, sgd, bias)
    elif ex:
        model_name = 'ex_model.h5'
        model = build_ex_model(int(np.max(x) + 1), int(np.max(y) + 1), dim, sgd)
    cb = [EarlyStopping(monitor='val_rmse', patience=30, verbose=1, mode='min'),
    TensorBoard(log_dir='./log', write_images=True),
    ModelCheckpoint(filepath=model_name, monitor='val_rmse', mode='min',save_best_only=True)]
    history = None
    if not whole:
        # if mf:
        history = model.fit([x, y],z, batch_size=bs, epochs=epo, validation_data=([vx,vy],vz), callbacks=cb, verbose=1)
        # elif ex:
        #     u,m = retrive_extra()
        #     history = model.fit([u, m, x, y],z, batch_size=bs, epochs=epo, validation_data=([vx,vy],vz), callbacks=cb, verbose=1)
    else:
        # if mf:
        history = model.fit([x, y],z, batch_size=bs, epochs=epo, callbacks=cb, verbose=1)
        # elif ex:
        #     u,m = retrive_extra()
        #     history = model.fit([u, m, x, y],z, batch_size=bs, epochs=epo, callbacks=cb, verbose=1)

    model.save('model.h5')
    return history

if __name__ == '__main__':
    train(ex=True)
    test()
    
    
