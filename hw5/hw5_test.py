from gensim.models import word2vec
import re, json
from math import floor
import numpy as np
from gensim import models
from keras.models import load_model
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import logging
from output import output
import sys, os, urllib

def load_test_data(testname='testing_data.txt',no_comma=False):
    test_data = open(testname).read()
    if no_comma:
        gex = re.compile("[\!,\.:;\?\'\"]")
        test_data = gex.sub(' ',test_data)
        gex = re.compile(' +')
        test_data = gex.sub(' ',test_data)
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
    x, maxlen = handle_type_err(word2idx, x)
    X_test = np.zeros((len(x),maxlen),dtype=int)
    for i in range(X_test.shape[0]):
        X_test[i][:x[i].shape[0]] = x[i]
    print("sentence max length:",maxlen)

    return X_test


def handle_type_err(word2idx, x):
    maxlen = 0
    gex = re.compile(r'(\w)(\1)+')
    gex0 = re.compile('(0)')
    for i in range(len(x)):
        for j in range(len(x[i])):
            try:
                x[i][j] = word2idx[x[i][j]]
            except:
                #print(x[i][j])
                noo = gex0.sub(r'o',x[i][j])
                new = gex.sub(r'\1',noo)
                if new in word2idx.keys():
                    # print(x[i][j],"->",new)
                    # saving_words.append(new)
                    x[i][j] = word2idx[new]
                else:
                    new = gex.sub(r'\1\1',noo)
                    if new in word2idx.keys():
                        # print(x[i][j],"->",new)
                        # saving_words.append(new)
                        x[i][j] = word2idx[new]
                    else:
                        # missing_words.append(x[i][j])
                        x[i][j] = 0
        x[i] = np.array(x[i],dtype = int)
        if len(x[i])>maxlen:
            maxlen = len(x[i])
    return x, maxlen

def test(testname='testing_data.txt',filename = "ans.csv",model_name='model.h5',no_comma=False):
    test = load_test_data(testname,no_comma=no_comma)
    ans = []
    model = load_model(model_name)
    result = np.argmax(model.predict(test, batch_size = 512, verbose=1),axis=1)
    for idx in range(result.shape[0]):
        ans.append([idx,result[idx]])
    output(ans,name=filename)

if __name__ == "__main__":
    if not os.path.exists('./ensemble_82966.h5'):
        _url = 'https://www.dropbox.com/s/zxgdldl6aiwxekw/ensemble_82966.h5?dl=1'
        urllib.urlretrieve(_url, "ensemble_82966.h5")
    test(testname=sys.argv[1], filename=sys.argv[2], model_name='ensemble_82966.h5')
    os.remove("ensemble_82966.h5")