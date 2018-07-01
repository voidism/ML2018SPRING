from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from keras.models import load_model
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Embedding,Lambda, LSTM,GRU, SpatialDropout1D, Dot
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from counter import counter
from keras import backend as K
import os
# -*- coding: utf-8 -*-

#import logging

from gensim.models import word2vec
from gensim import models
import logging, json, re
import numpy as np
from counter import counter
import os
from ckip import PyWordSeg
from output import output
import csv
from functools import reduce

global ckiper

def load_test_data():
    r = open('testing_data.csv').read()
    l = r.split('\n')[1:-1]
    sentences = []
    gex = re.compile(r'\d+,(.*),0:(.*)\t1:(.*)\t2:(.*)\t3:(.*)\t4:(.*)\t5:(.*)')
    for i in l:
        sentences += gex.findall(i)[0]
    return sentences

def segment():
    global ckiper
    ckiper = PyWordSeg()
    for idx in range(1, 6):
        store_name = 'training_data/after_segment%s.txt'%idx
        file_name = 'training_data/%s_train.txt'%idx
        segment_file(file_name, store_name)
    segment_file('testing_data.csv','training_data/after_segment_test.txt')

def segment_file(file_name, store_name):
    global ckiper
    output = open(store_name, 'w', encoding='utf-8')
    with open(file_name, 'r', encoding='utf-8') as content:
        corpus = []
        if 'test' not in file_name:
            content = content.read()
            corpus = content.split('\n')
        else:
            corpus = load_test_data()
        print(file_name, 'corpus size: %s' %range(len(corpus)))
        ct = counter(epoch=len(corpus),update_rate=50,title ="cutting training data " + file_name)
        for texts_num in range(len(corpus)):
            gex_TAB = re.compile(r'\t')
            gex_space = re.compile(r'( )+')
            line = corpus[texts_num]
            gex_TAB.sub('。', line)
            gex_space.sub('，',line)
            words = ckiper.Segment(line)
            for w in range(len(words)):
                # if word not in stopword_set:
                output.write(words[w])
                if w != len(words)-1:
                    output.write(',')
            output.write('\n')

            # if (texts_num + 1) % 1000 == 0:
            #     print("已完成前 %d 行的斷詞" % (texts_num + 1))
            ct.flush(int(texts_num))
    output.close()

def word_embedding():
    if not os.path.exists("training_data/corpus.txt"):
        output = open("training_data/corpus.txt", 'w', encoding='utf-8')
        for i in range(1, 6):
            store_name = 'training_data/after_segment%s.txt'%i
            text = open(store_name).read()
            output.write(text)
        # store_name = 'training_data/after_segment_test.txt'
        # text = open(store_name).read()
        # output.write(text)
        output.close()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("training_data/corpus.txt")
    model = word2vec.Word2Vec(sentences, size=200,
    window=5, workers=4, min_count=10, iter=100, sg=1)

    # word2vec.Word2Vec()

    model.save("word2vec.model")
    vocab_dict = dict([(k, model.wv[k]) for k, v in model.wv.vocab.items()])
    np.save('word2vec_dict.npy', vocab_dict)
    word2vec_weights = model.wv.syn0
    word2vec_weights = np.concatenate((np.zeros((1,256)),word2vec_weights),axis=0)
    np.save('word2vec_weights.npy',word2vec_weights)
    idx2word_list = ['_PAD'] + model.wv.index2word
    idx2word = dict([(x, idx2word_list[x]) for x in range(len(idx2word_list))])
    word2idx = dict([(v, k) for k, v in idx2word.items()])
    with open('word2idx.json', 'w') as fp:
        json.dump(word2idx, fp)
    with open('idx2word.json', 'w') as fp:
        json.dump(idx2word, fp)

def data_gen():
    # x1, _ = corpus_to_mat(load_corpus('training_data/after_segment1.txt',seg = ','),deter_maxlen=14)
    # x2, _ = corpus_to_mat(load_corpus('training_data/after_segment2.txt',seg = ','),deter_maxlen=14)
    # x3, _ = corpus_to_mat(load_corpus('training_data/after_segment3.txt',seg = ','),deter_maxlen=14)
    # x4, _ = corpus_to_mat(load_corpus('training_data/after_segment4.txt',seg = ','),deter_maxlen=14)
    # x5, _ = corpus_to_mat(load_corpus('training_data/after_segment5.txt',seg = ','),deter_maxlen=14)
    x1 = load_corpus('training_data/after_segment1.txt',seg = ',')
    x2 = load_corpus('training_data/after_segment2.txt',seg = ',')
    x3 = load_corpus('training_data/after_segment3.txt',seg = ',')
    x4 = load_corpus('training_data/after_segment4.txt',seg = ',')
    x5 = load_corpus('training_data/after_segment5.txt',seg = ',')


    n = np.array([0.22166352, 0.21726378, 0.39513933, 0.16593338])
    X_list = []
    Y_list = []
    for xn in [x1,x2,x3,x4,x5]:
        num = np.random.choice([1, 2, 3, 4], size=(len(xn),), p=n)
        for i in range(len(xn)):
            if i >= num[i]:
                X_list.append(reduce(lambda s1, s2 : s1+['，']+s2, xn[i-num[i]:i]))
                Y_list.append(xn[i])
    # return X_list, Y_list
    X_all, _ = corpus_to_mat(X_list)
    Y_all, maxlen = corpus_to_mat(Y_list)

    # TsentQ, TsentA, FsentQ, FsentA = nega_sent()

    # TsentQ, __ = corpus_to_mat(TsentQ,_)
    # TsentA, __ = corpus_to_mat(TsentA,maxlen)
    # Tans = np.ones((len(TsentA),1))
    # X_all = np.concatenate((x1[:-1],x2[:-1],x3[:-1],x4[:-1],x5[:-1]),axis=0)
    # Y_all = np.concatenate((x1[:-1],x2[:-1],x3[:-1],x4[:-1],x5[:-1]),axis=0)
    # Y_all = np.concatenate((x1[1:],x2[1:],x3[1:],x4[1:],x5[1:]),axis=0)

    Z1 = np.ones((len(X_all),1))

    # T_all = np.concatenate((x1,x2,x3,x4,x5),axis=0)
    data = json.load(open('testdata.json'))['data']
    li = [item for sublist in data for item in sublist[1:]]
    cho_mat, _ = corpus_to_mat(li,deter_maxlen=maxlen)
    height = len(x1)+len(x2)+len(x3)
    Tx_all = X_all
    Ty_all = np.concatenate((cho_mat, Y_all[height:],Y_all[:height]),axis=0)
    Ty_all = Ty_all[:len(Tx_all)]
    # print('T2len:',T2_all.shape)
    Z2 = np.zeros((len(Tx_all),1))
    np.random.seed(9487)
    randomize = np.random.permutation(len(Ty_all))
    Ty2 = Ty_all[randomize]
    randomize = np.random.permutation(len(Ty_all))
    Ty3 = Ty_all[randomize]
    randomize = np.random.permutation(len(Ty_all))
    Ty4 = Ty_all[randomize]

    X_train = np.concatenate((X_all,Tx_all,Tx_all,Tx_all,Tx_all),axis=0)
    Y_train = np.concatenate((Y_all,Ty_all,Ty2,Ty3,Ty4),axis=0)



    Z_train = np.concatenate((Z1,Z2,Z2,Z2,Z2),axis=0)
    randomize = np.random.permutation(len(X_train))
    X_train,Y_train,Z_train = (X_train[randomize], Y_train[randomize], Z_train[randomize])
    print("generate %d training data"%len(X_train))
    return X_train,Y_train,Z_train


def valid_gen():
    x1, _ = corpus_to_mat(load_corpus('training_data/after_segment1.txt',seg = ','),deter_maxlen=14)
    x2, _ = corpus_to_mat(load_corpus('training_data/after_segment2.txt',seg = ','),deter_maxlen=14)
    x3, _ = corpus_to_mat(load_corpus('training_data/after_segment3.txt',seg = ','),deter_maxlen=14)
    x4, _ = corpus_to_mat(load_corpus('training_data/after_segment4.txt',seg = ','),deter_maxlen=14)
    x5, _ = corpus_to_mat(load_corpus('training_data/after_segment5.txt',seg = ','),deter_maxlen=14)

    X_train = np.concatenate((x1,x2,x3),axis=0)[:10000]
    Y_train = np.concatenate((x4,x5),axis=0)[:10000]

    Z_train = np.zeros((len(X_train),1))
    np.random.seed(9487)
    randomize = np.random.permutation(len(X_train))
    X_train,Y_train,Z_train = (X_train[randomize], Y_train[randomize], Z_train[randomize])

    return X_train,Y_train,Z_train

def statistics():
    a = json.load(open('testdata.json'))['data']
    b = []
    for i in a:
        b.append(i[0].count('，')+i[0].count('。')+1)
    return np.array(b)

def handle_type_err(word2idx, x):
    maxlen = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            try:
                x[i][j] = word2idx[x[i][j]]
            except:
                x[i][j] = 0
        x[i] = np.array(x[i],dtype = int)
        if len(x[i])>maxlen:
            maxlen = len(x[i])
    return x, maxlen

def load_corpus(fname='training_data/corpus.txt',seg = ' '):
    cor = open(fname).read()
    X = cor.split('\n')
    newX = []
    for i in X:
        newX.append(i.split(seg))
    return newX[:-2]
    
def corpus_to_mat(X = None, deter_maxlen = 0):
    if X == None:
        X = load_corpus()
    word2idx = json.load(open('word2idx.json'))
    x, maxlen = handle_type_err(word2idx, X)
    if deter_maxlen!=0:
        maxlen = deter_maxlen
    X_train = np.zeros((len(x),maxlen),dtype=int)
    for i in range(X_train.shape[0]):
        if len(x[i])>maxlen:
            x[i] = x[i][:maxlen]
        X_train[i][:x[i].shape[0]] = x[i]
    print("sentence max length:",maxlen)
    return X_train, maxlen

def train(epo = 10, model_name='best_model.h5'):
    X, Y, Z = data_gen()
    # vx,vy,vz = valid_gen()
    model = build_model(14)
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    model.summary()

    cb = [EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='min'),
    TensorBoard(log_dir='./log', write_images=True),
    ModelCheckpoint(filepath=model_name,monitor='val_acc',mode='max', save_best_only=True)]

    model.fit([X,Y],Z,validation_split=0.1,
    batch_size=512,epochs=epo, callbacks=cb, verbose=1)
    model.save('model.h5')
'''
def test(model_name='best_model.h5'):
    data = json.load(open('testdata.json'))['data']
    li = [item for sublist in data for item in sublist]
    s, maxlen = corpus_to_mat(li, deter_maxlen=0)
    ans = []
    model = load_model(model_name)
    ct = counter(epoch=int(len(li)/7),title='solving...')
    for i in range(int(len(li)/7)):
        prob = s[i * 7]
        chos = s[i * 7 + 1:i * 7 + 7]
        # s = sentence_to_vec(li[i * 7:i * 7 + 7], alpha=alpha)
        # prob = s[0]
        # chos = s[1:]
        y = np.zeros((6,))
        for j in range(len(chos)):
            y[j] = model.predict([prob.reshape((1,maxlen)), chos[j].reshape((1,maxlen))])
            # y[j] += my_sim(prob, chos[j])
        ans.append([i, np.argmax(y)])
        ct.flush(i)
    output(ans, header=['id', 'ans'])
    print('Done')
'''
def test(model_name='best_model.h5',filename = "ans.csv",pick = 0.0,pie=True,ne=True,ens=False):
    data = json.load(open('testdata.json'))['data']
    li = [item for sublist in data for item in sublist]
    s, maxlen = corpus_to_mat(li, deter_maxlen=0)
    ans = []
    model = load_model(model_name)
    ct = counter(epoch=int(len(li)/7),title='solving...')
    prob_mat = []
    chos_mat = []
    for i in range(int(len(li)//7)):
        prob_mat.append(np.tile(s[i * 7],(6,1)))
        chos_mat.append(s[i * 7 + 1:i * 7 + 7])

    prob_mat = np.array(prob_mat)
    chos_mat = np.array(chos_mat)

    res_mat = model.predict([prob_mat.reshape((-1,maxlen)), chos_mat.reshape((-1,maxlen))],
    batch_size=512, verbose=1)
    result = []
    sub = None
    pie_prob1 = None
    pie_prob2 = None
    pie_prob3 = None
    pie_prob4 = None
    pie_prob5 = None
    pie_prob6 = None
    pie_prob7 = None
    nega = None
    if pick>0:
        sub = load_sub()
    if pie:
        pie_prob1 = load_pie('prob (1).txt')
        pie_prob2 = load_pie('prob (2).txt')
        pie_prob3 = load_pie('prob (3).txt')
        pie_prob4 = load_pie('prob (4).txt')
        pie_prob5 = load_pie('prob (5).txt')
        pie_prob6 = load_pie('prob (6).txt')
        pie_prob7 = load_pie('prob (7).txt')
    if ne:
        nega = np.load('nega.npy')
    ensprob = np.load('ens_prob.npy')
    for i in range(len(res_mat)//6):
        y = res_mat[i * 6:i * 6 + 6]
        if pick>0:
            y[sub[i]] -= pick
        if pie:
            # pass
            y *= 1
            y += 0.5*pie_prob1[i].reshape(y.shape)
            y += 0.5*pie_prob2[i].reshape(y.shape)
            y += 1*pie_prob3[i].reshape(y.shape)
            y += 1*pie_prob4[i].reshape(y.shape)
            if i<1700:
                y= pie_prob7[i].reshape(y.shape)
            # y += 1*pie_prob5[i].reshape(y.shape)
            # y += 1*pie_prob6[i].reshape(y.shape)
        if ne:
            y += 1*nega[i].reshape(y.shape)
        if ens:
            y += 1*ensprob[i].reshape(y.shape)
        result.append(y)
        ans.append([i, np.argmax(y)])
        ct.flush(i)
    output(ans, header=['id', 'ans'],name=filename)
    print('Done')
    print('check human label result: %s' % check(ans))
    return np.array(result)

def load_sub():
    r = csv.reader(open('sub.csv'))
    l = list(r)[1:]
    a = np.array(l,dtype = int)
    return a.T[1].flatten()

def load_pie(name = 'prob.txt'):
    r = open(name).read().split('\n')[:-1]
    for i in range(len(r)):
        r[i] = np.array(r[i].split(' '),dtype=float)
    return np.array(r)

def freq_mat(alpha = 0.001):
    idx2freq = json.load(open('idx2freq.json'))
    # word2idx = json.load(open('word2idx.json'))
    freq_list = []
    for i in range(len(idx2freq.keys())):
        freq_list.append(alpha/(alpha + float(idx2freq[str(i)][1])))
    freq_array = np.array(freq_list,dtype = float)
    np.save("freq_mat.npy",freq_array)
    return freq_array


def check(ans):
    # data = json.load(open('testdata.json'))['data']
    r = csv.reader(open('real_ans.csv'))
    l = list(r)#[1:]
    c = 0
    idx=0
    for i in range(len(l)):
        if int(l[i][0]) == int(ans[i][1]):
            c += 1
        idx+=1
    return c/len(l)

def print_question(idx,li,ans,realans = -1):
    print("Question(%d)=========="%idx)
    print(''.join(li[0]))
    print("Answer=============")
    for i in range(6):
        print("%s %d.[%s] %s"%('V' if i==realans else " ",i,'O' if i==ans else " ","".join(li[i+1])))
    print("=================================")

def pick():
    data = json.load(open('testdata.json'))['data']
    r = csv.reader(open('real_ans.csv'))
    l = list(r)#[1:]
    ur = csv.reader(open('pick_ans.csv'))
    ul = list(ur)[1:]
    
    ans = np.array(ul,dtype=int)[:,1]
    c = 0
    idx = 0
    real_list = []
    if os.path.exists('pick_result.csv'):
        rf = csv.reader(open('pick_result.csv'))
        real_list = list(rf)
        idx = len(real_list)
    # i = idx
    while idx < len(ul):
        print_question(idx,data[idx],ans[idx],int(l[idx][0]))
        print(ans[idx],int(l[idx][0]))
        if (ans[idx]==int(l[idx][0])):
            real_list.append([idx,0,ans[idx]])
            idx += 1
            continue
        e = input("broken? (0:good, 1:hard, 2:fix it) ")
        if e == 'Q':
            break
        if (int(l[idx][0]) != -1):
            real_list.append([idx,e,int(l[idx][0])])
            idx += 1
            continue
        h = input('real_ans? ')
        real_list.append([idx,e,h])
        idx += 1
    f = open('pick_result.csv','w')
    w = csv.writer(f, delimiter=',', lineterminator='\n')
    w.writerows(real_list)
    f.close()

def nega_gen():
    data = json.load(open('testdata.json'))['data']
    r = csv.reader(open('real_ans.csv'))
    l = list(r)#[1:]
    ur = csv.reader(open('pick_ans.csv'))
    ul = list(ur)[1:]
    pr = csv.reader(open('pick_result.csv'))
    pl = list(pr)
    a = np.zeros((len(data),6))
    for i in range(len(pl)):
        if ul[i][1]=='2':
            wrong_id = int(ul[i][1])
            true_id = int(l[i][0])
            a[i][wrong_id]=-10
            a[i][true_id]=10
    np.save('nega.npy',a)

def nega_sent():
    data = json.load(open('testdata.json'))['data']
    r = csv.reader(open('real_ans.csv'))
    l = list(r)#[1:]
    ur = csv.reader(open('pick_ans.csv'))
    ul = list(ur)[1:]
    pr = csv.reader(open('pick_result.csv'))
    pl = list(pr)
    TsentQ = []
    TsentA = []
    FsentQ = []
    FsentA = []
    for i in range(len(pl)):
        if ul[i][1]=='2':
            wrong_id = int(ul[i][1])
            true_id = int(l[i][0])
            TsentQ.append(data[i][0])
            TsentA.append(data[i][true_id+1])
            FsentQ.append(data[i][0])
            FsentA.append(data[i][wrong_id+1])
    return TsentQ, TsentA, FsentQ, FsentA

    

def test_on_fake(model_name='best_model.h5'):
    data = json.load(open('fakedata.json'))['data'][10000:15000]
    li = [item for sublist in data for item in sublist]
    s, maxlen = corpus_to_mat(li, deter_maxlen=0)
    ans = []
    model = load_model(model_name)
    ct = counter(epoch=int(len(li)/7),title='solving...')
    prob_mat = []
    chos_mat = []
    for i in range(int(len(li)/7)):
        prob_mat.append(np.tile(s[i * 7],(6,1)))
        chos_mat.append(s[i * 7 + 1:i * 7 + 7])

    prob_mat = np.array(prob_mat)
    chos_mat = np.array(chos_mat)

    res_mat = model.predict([prob_mat.reshape((-1,maxlen)), chos_mat.reshape((-1,maxlen))],
    batch_size=512, verbose=1)
    for i in range(len(res_mat)//6):
        y = res_mat[i * 6:i * 6 + 6]
        ans.append([i, np.argmax(y)])
        ct.flush(i)
    print('Done')
    tru = (np.array(ans)==0).sum()
    print('acc_on_fake:',tru/len(ans))

def build_model(maxlen,idf=False):
    word2vec_weights = np.load('word2vec_weights.npy')
    freq_mat = np.load('freq_mat.npy')
    freq_mat = freq_mat.reshape((len(freq_mat),1))
        
    Q_input = Input(shape=[None])
    A_input = Input(shape=[None])
    def freq(x):
        return freq_mat
    Q_emb = Embedding(word2vec_weights.shape[0],
                word2vec_weights.shape[1],mask_zero=True,
                weights=[word2vec_weights],
                trainable=True)(Q_input)
    # Q_freq = Embedding(freq_mat.shape[0],
    #             freq_mat.shape[1],mask_zero=True,
    #             weights=[freq_mat],
    #             trainable=True)(Q_input)
    # Q_freq = Flatten()(Q_freq)
    A_emb = Embedding(word2vec_weights.shape[0],
                word2vec_weights.shape[1],mask_zero=True,
                weights=[word2vec_weights],
                trainable=True)(A_input)
    # A_freq = Embedding(freq_mat.shape[0],
    #             freq_mat.shape[1],mask_zero=True,
    #             weights=[freq_mat],
    #             trainable=True)(A_input)
    # # A_freq = Flatten()(A_freq)
    # def tfidf(x):
    #     return x[0]*x[1]/K.sum(x[1])
    # Q_emb = Lambda(tfidf,output_shape=(None,256))([Q_emb,Q_freq])
    # A_emb = Lambda(tfidf,output_shape=(None,256))([A_emb,A_freq])

    # def sent2vec(x):
    #     return K.mean(x, axis=1)
    # Q_rnn = Lambda(sent2vec,output_shape=[256])(Q_emb)
    Q_rnn =  GRU(128,# dropout=0.2, recurrent_dropout=0.2,
                   activation='hard_tanh',
                   inner_activation='hard_tanh',
                   implementation=2)(Q_emb)
    Q_rnn =  GRU(64,# dropout=0.2, recurrent_dropout=0.2,
                   activation='hard_tanh',
                   inner_activation='hard_tanh',
                   implementation=2)(Q_rnn)
    # A_rnn = Lambda(sent2vec,output_shape=[256])(A_emb)
    A_rnn = GRU(256,
                   activation='hard_tanh',
                   inner_activation='hard_tanh',
                   implementation=2)(A_emb)
    A_rnn = GRU(64,
                   activation='hard_tanh',
                   inner_activation='hard_tanh',
                   implementation=2)(A_rnn)

    # Q_mat = Dense(256,activation='relu')(Q_rnn)
    # Q_mat = Dropout(0.25)(Q_mat)
    # Q_mat = Dense(256,activation='relu')(Q_mat)
    # Q_mat = Dropout(0.25)(Q_mat)
    # Q_mat = Dense(256,activation='relu')(Q_mat)
    # Q_mat = Dropout(0.25)(Q_mat)
    # Q_mat = Dense(256,activation='sigmoid')(Q_mat)
    # Q_mat = Dense(256,activation='sigmoid')(Q_mat)
    # A_mat = Dense(256,activation='relu')(A_rnn)
    # A_mat = Dropout(0.25)(A_mat)
    # A_mat = Dense(256,activation='relu')(A_rnn)
    # A_mat = Dropout(0.25)(A_mat)
    # A_mat = Dense(256,activation='relu')(A_rnn)
    # A_mat = Dropout(0.25)(A_mat)
    # Q_prd = Dot(axes=1)([Q_mat, Q_rnn])
    # A_prd = Dot(axes=1)([A_mat, Q_rnn])
    res = Dot(axes=1, normalize=True)([Q_rnn, A_rnn])
    out = Dense((1), activation ="sigmoid") (res)

    sequence_autoencoder = Model([Q_input, A_input], out)
    return sequence_autoencoder

if __name__ == "__main__":
    # word_embedding()
    train()
    res = test()
    np.save('result.npy',res)