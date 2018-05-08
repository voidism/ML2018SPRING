from gensim.models import word2vec
import re, json
from math import floor
import numpy as np
from gensim import models
from keras.models import load_model
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
import logging
from output import output

def load_word_data():
    label_data = open('training_label.txt').read()
    nobel_data = open('training_nolabel.txt').read()
    test_data = open('testing_data.txt').read()
    a = label_data.split('\n')
    b = nobel_data.split('\n')
    c = test_data.split('\n')

    for i in range(len(a)):
        a[i] = a[i][10:]

    c = c[1:-1]
    gex = re.compile(r'(\d*),(.*)')
    for i in range(len(c)):
        try:
            c[i] = gex.findall(c[i])[0][1]
        except:
            print(i, c[i])
    all = a + b + c
    ret = []
    for i in all:
        ret.append(i.split(' '))
    return ret

def load_train_data():
    label_data = open('training_label.txt').read()
    a = label_data.split('\n')[:-1]
    y = []

    for i in range(len(a)):
        y.append(int(a[i][0]))
        a[i] = a[i][10:]

    x = []
    for i in a:
        x.append(i.split(' '))

    word2idx = json.load(open('word2idx.json'))
    maxlen = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            try:
                x[i][j] = word2idx[x[i][j]]
            except:
                print('no such word:',x[i][j])
                x[i][j] = 0
        x[i] = np.array(x[i],dtype = int)
        if len(x[i])>maxlen:
            maxlen = len(x[i])
    X_train = np.zeros((len(x),maxlen),dtype=int)
    for i in range(X_train.shape[0]):
        X_train[i][:x[i].shape[0]] = x[i]
    print("sentence max length:",maxlen)
    y = np.array(y,dtype=int)

    # one-hot encoding
    Y_train = np.zeros((y.shape[0],2))
    Y_train[np.arange(y.shape[0]), y] = 1

    return X_train, Y_train

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

# def word2idx(word_model, word):
#     return word_model.wv.vocab[word].index

# def idx2word(word_model, idx):
#     return word_model.wv.index2word[idx]


def train_word2vec():
    sentences = load_word_data()
    model = word2vec.Word2Vec(sentences, size=250)
    model.save("word2vec.model")
    vocab_dict = dict([(k, model.wv[k]) for k, v in model.wv.vocab.items()])
    np.save('word2vec_dict.npy', vocab_dict)
    word2vec_weights = model.wv.syn0
    word2vec_weights = np.concatenate((np.zeros(1,250),word2vec_weights),axis=1)
    np.save('word2vec_weights.npy',word2vec_weights)
    idx2word_list = ['_PAD'] + model.wv.index2word
    idx2word = dict([(x, idx2word_list[x]) for x in range(len(idx2word_list))])
    word2idx = dict([(v, k) for k, v in idx2word.items()])
    with open('word2idx.json', 'w') as fp:
        json.dump(word2idx, fp)
    with open('idx2word.json', 'w') as fp:
        json.dump(idx2word, fp)

def build_model(word2vec_weights):
    model = Sequential()
    model.add(Embedding(word2vec_weights.shape[0],
                word2vec_weights.shape[1],mask_zero=True,
                weights=[word2vec_weights],
                trainable=False))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.summary()
    return model

def train_rnn():
    word2vec_weights = np.load('word2vec_weights.npy')
    # word2vec_dict = json.load('word2vec.json')
    # word2vec_model = models.Word2Vec.load('word2vec.model')

    model = build_model(word2vec_weights)
    x, y = load_train_data()
    X_train, Y_train, X_test, Y_test = split_valid(x, y)
    model.fit(X_train, Y_train, batch_size=256,epochs=40,
              validation_data=(X_test, Y_test))
    model.save('model.h5')

def load_test_data(testname):
    test_data = open(testname).read()
    c = test_data.split('\n')

    c = c[1:-1]
    gex = re.compile(r'(\d*),(.*)')
    for i in range(len(c)):
        try:
            c[i] = gex.findall(c[i])[0][1]
        except:
            print(i, c[i])

    x = []
    for i in c:
        x.append(i.split(' '))

    word2idx = json.load(open('word2idx.json'))
    maxlen = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            try:
                x[i][j] = word2idx[x[i][j]]
            except:
                print('no such word:',x[i][j])
                x[i][j] = 0
        x[i] = np.array(x[i],dtype = int)
        if len(x[i])>maxlen:
            maxlen = len(x[i])
    X_test = np.zeros((len(x),maxlen),dtype=int)
    for i in range(X_test.shape[0]):
        X_test[i][:x[i].shape[0]] = x[i]
    print("sentence max length:",maxlen)

    return X_test

def test(testname='testing_data.txt',filename = "ans.csv",model_name='model.h5'):
    test = load_test_data(testname)
    ans = []
    model = load_model('model.h5')
    result = np.argmax(model.predict(test),axis=1)
    for idx in range(result.shape[0]):
        ans.append([idx,result[idx]])
    output(ans,name=filename)


if __name__ == "__main__":
    logging.basicConfig(format=chr(13)+'%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # train_word2vec()
    train_rnn()
    test()