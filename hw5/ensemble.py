from keras.models import load_model
from keras import layers
from keras.models import Model
from keras.layers import Input
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

def ensemble_models(model_collection):
    model_input = Input(shape=model_collection[0].input_shape[1:])
    pre_avg=[model(model_input) for model in model_collection] 
    Avg=layers.average(pre_avg)
    modelEns = Model(inputs=model_input, outputs=Avg, name='ensemble')
    modelEns.save('ensemble.h5')
    return modelEns

def test(testname='testing_data.txt',filename = "ans.csv",model_name='model.h5'):
    test = load_test_data(testname)
    ans = []
    model = load_model(model_name)
    result = np.argmax(model.predict(test),axis=1)
    for idx in range(result.shape[0]):
        ans.append([idx,result[idx]])
    output(ans,name=filename)


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
    # missing_words = []
    gex = re.compile(r'(\w)(\1)+')
    gex0 = re.compile('(0)')
    for i in range(len(x)):
        for j in range(len(x[i])):
            try:
                x[i][j] = word2idx[x[i][j]]
            except:
                # missing_words.append(x[i][j])
                #print(x[i][j])
                new = gex0.sub(r'o',x[i][j])
                new = gex.sub(r'\1',new)
                if new in word2idx.keys():
                    # print(x[i][j],"->",new)
                    x[i][j] = word2idx[new]
                else:
                    x[i][j] = 0
        x[i] = np.array(x[i],dtype = int)
        if len(x[i])>maxlen:
            maxlen = len(x[i])
    # np.save('missing.npy',missing_words)
    X_test = np.zeros((len(x),maxlen),dtype=int)
    for i in range(X_test.shape[0]):
        X_test[i][:x[i].shape[0]] = x[i]
    print("sentence max length:",maxlen)

    return X_test

if __name__ == "__main__":
    model_collection = []
    models=['model_82409.h5','model_82671.h5','model_82671.h5','model_82296.h5']
    for i in models:
        model_collection.append(load_model(i))
    ensemble_models(model_collection)
    test(testname='testing_data.txt',filename = "ens_ans.csv",model_name='ensemble.h5')
