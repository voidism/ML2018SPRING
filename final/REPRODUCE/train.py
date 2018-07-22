import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.functional import pad
from torch.nn.functional import avg_pool3d
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import sys
import re
import gensim
from gensim.models.word2vec import Word2Vec
import io
import random

#settings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#self defined function
class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
    def forward(self,x):
        return x*functional.sigmoid(x)

#get dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self,df,loader=torchvision.datasets.folder.default_loader,task=None,maxlength=20,usingw2v=False,trainw2v=False,w2vwindow=2,embed_size=512,dictionary=None):
        self.task = task
        self.loader = loader
        self.maxlength = maxlength
        self.usingw2v = usingw2v
        self.trainw2v = trainw2v
        self.w2vwindow = w2vwindow
        self.embed_size = embed_size
        self.dictionary = dictionary
        if self.task is 'train':
            self.label = []
            self.df = []
            for i in df:
                self.label.append(random.randint(0,5))
                self.df.append(list(map(str,i.split(b' '))))
            if self.usingw2v is False:
                self.makedict()
                self.dictionary = self.Tok2Ind
                self.substitute(self.dictionary)
        elif self.task is 'vald':
            self.label = []
            self.df = []
            for i in df:
                self.label.append(random.randint(0,5))
                self.df.append(list(map(str,i.split(b' '))))
            if self.usingw2v is False:
                self.substitute(self.dictionary)
        elif self.task is 'extra':
            self.df = []
            for i in df:
                self.df.append(list(map(str,i.split(b' '))))
            if self.usingw2v is False:
                self.substitute(self.dictionary)
        elif self.task is 'test':
            self.df = []
            for i in df:
               i = i.split(b',')
               self.df.append([list(map(str,i[1].split(b' ')))])
               i = i[2].split(b':')
               for j in range(1,len(i)-1):
                   self.df[-1].append(list(map(str,i[j][:-1].split(b' '))))
               self.df[-1].append(list(map(str,i[len(i)-1].split(b' '))))
            if self.usingw2v is False:
                self.substitute(self.dictionary)
        elif self.task is None:
            raise Exception('Error : No task specified for dataloader')
        else:
            raise Exception('Error : Unknown Task')

    def sampleline(self,lines):
        if self.usingw2v is False:
            vec = torch.LongTensor(lines)
        if self.usingw2v is True:
            vec = []
            for idx in range(len(lines)):
                vec.append([])
                length = len(lines[idx])
                for i,word in enumerate(lines[idx]):
                    if word in self.dictionary.vocab:
                        vec[-1].append(self.dictionary[word].tolist())
                    else:
                        vec[-1].append([0 for leng in range(self.embed_size)])
                vec[-1] = vec[-1][max(0,len(vec[-1])-self.maxlength):]
                vec[-1] = [[0 for leng in range(self.embed_size)] for padnum in range(self.maxlength-length)]+vec[-1]
                #print(lines[idx])
                vec[-1] = torch.FloatTensor(vec[-1])
        return vec

    def __getitem__(self,index):
        if self.task is 'train' or self.task is 'vald':
            line = []
            for i in range(7):
                line.append(self.df[random.randint(0,len(self.df)-1)])
            line[0] = self.df[index]
            line[self.label[index]+1] = self.df[index+1]
            return self.sampleline(line),self.label[index]
        elif self.task is 'extra':
            line = self.df[index]
            return self.sampleline(line)
        elif self.task is 'test':
            line = self.df[index]
            return self.sampleline(line)
        elif self.task is None:
            raise Exception('Error : No task specified for dataloader')
        else:
            raise Exception('Error : Unknown Task')

    def __len__(self):
        n=len(self.df)
        if self.task is 'train' or self.task is 'vald':
            return n-1
        else:
            return n

    def makedict(self):
        self.Tok2Ind = {'<PAD>':0,'<BOS>':1,'<EOS>':2,'<UNK>':3}
        self.Ind2Tok = {0:'<PAD>',1:'<BOS>',2:'<EOS>',3:'<UNK>'}
        self.Count = {}
        for line in self.df:
            for word in line:
                if word not in self.Count.keys():
                    self.Count[word] = 0
                self.Count[word]+=1
        tmp = sorted(self.Count.items(),key=lambda x:x[1],reverse=True)
        collect_words = len(tmp)
        for item,index in zip(tmp,list(range(4,collect_words+4))):
            self.Tok2Ind[item[0]] = index
            self.Ind2Tok[index] = item[0]

#NN Model
class GRUencoder(nn.Module):
    def __init__(self,hidden_size,batch_size,embed_size,dict_size,usingw2v,maxlength,usingswish):
        super(GRUencoder,self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.usingw2v = usingw2v
        self.layers = 2
        self.embed = nn.Embedding(self.dict_size,self.embed_size)
        self.gru = nn.GRU(self.embed_size,self.hidden_size,self.layers,batch_first=True,dropout=0.0)

    def forward(self,ques,q1,q2,q3,q4,q5,q6):
        if self.usingw2v is False:
            ques = self.embed(ques)
        out0,hid = self.gru(ques)
        out1,hid = self.gru(q1)
        out2,hid = self.gru(q2)
        out3,hid = self.gru(q3)
        out4,hid = self.gru(q4)
        out5,hid = self.gru(q5)
        out6,hid = self.gru(q6)
        out0 = out0[:,-1,:]
        out1 = out1[:,-1,:]
        out2 = out2[:,-1,:]
        out3 = out3[:,-1,:]
        out4 = out4[:,-1,:]
        out5 = out5[:,-1,:]
        out6 = out6[:,-1,:]
        return torch.cat((out0,out1,out2,out3,out4,out5,out6),1)

class Classifier(nn.Module):
    def __init__(self,encoder,hidden_size,batch_size,embed_size,dict_size,maxlength,usingw2v,usingswish):
        super(Classifier,self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.dict_size = dict_size
        self.maxlength = maxlength
        self.usingw2v = usingw2v
        self.usingswish = usingswish
        self.seq2vec = encoder(self.hidden_size,self.batch_size,self.embed_size,self.dict_size,usingw2v,self.maxlength,usingswish)
        self.fc1 = nn.Linear(self.hidden_size*7,100)
        self.fc2 = nn.Linear(100,6)
        if self.usingswish is True:
            self.activation = Swish()
        else:
            self.activation = nn.RReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self,ques,q1,q2,q3,q4,q5,q6):
        encoded = self.seq2vec(ques,q1,q2,q3,q4,q5,q6)
        out = self.fc1(encoded)
        out = self.activation(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        out = functional.softmax(out,dim=1)
        return out

class Frame():
    def __init__(self,traindata=None,extradata=None,valddata=None,testdata=None,maxlength=41,batch_size=100,hidden_size=512,embed_size=128,num_epochs=20,learning_rate=1e-4,usingw2v=False,trainw2v=False,w2vwindow=2,usingswish=False,cuda=False,Log=None,loaddictpath=None,savedictpath=None,resultpath=None):
        self.maxlength = maxlength
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.usingw2v = usingw2v
        self.trainw2v = trainw2v
        self.w2vwindow = w2vwindow
        self.usingswish= usingswish
        self.cuda = cuda
        if Log is None:
            self.Log = sys.stdout
        else:
            self.Log = open(Log,'w')
        self.traindata = traindata
        self.extradata = extradata
        self.valddata = valddata
        self.testdata = testdata
        self.savedictpath = savedictpath
        self.loaddictpath = loaddictpath
        if resultpath is None:
            raise Exception('Error : Resultpath not specified')
        self.result = open(resultpath,'w')

    def loadtraindata(self):
        self.train_dataset, self.trainloader = self.loaddata(self.traindata,'train',False,None)

    def loadextradata(self,dictionary):
        self.extra_dataset, self.extraloader = self.loaddata(self.extradata,'extra',False,dictionary)

    def loadvalddata(self,dictionary):
        self.vald_dataset, self.valdloader = self.loaddata(self.valddata,'vald',False,dictionary)

    def loadtestdata(self,dictionary):
        self.test_dataset, self.testloader = self.loaddata(self.testdata,'test',False,dictionary)

    def loaddata(self,path=None,task=None,Shuffle=False,dictionary=None):
        if path is None:
            raise Exception('Error : Datapath not specified')
        fetch = open(path,'rb').read().split(b'\n')[:-1]
        if task is 'test':
            fetch = fetch[1:]
        dataset = Dataset(df=fetch,
                          task=task,
                          maxlength = self.maxlength,
                          usingw2v = self.usingw2v,
                          trainw2v = self.trainw2v,
                          w2vwindow = self.w2vwindow,
                          embed_size = self.embed_size,
                          dictionary = dictionary)
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch_size,
                                             shuffle=Shuffle,
                                             num_workers=0)
        return dataset,loader

    def trainword2vec(self,data):
        word2vec = Word2Vec(data,size=self.embed_size,window=self.w2vwindow,min_count=0,workers=8,seed=0)
        word2vec.save('WORD2VECDICT')
        del word2vec

    def loadword2vec(self):
        self.dictionary = Word2Vec.load('WORD2VECDICT').wv
        if self.traindata is not None:
            self.train_dataset.dictionary = self.dictionary
        if self.extradata is not None:
            self.train_dataset.dictionary = self.dictionary
        if self.valddata is not None:
            self.vald_dataset.dictionary = self.dictionary
        if self.testdata is not None:
            self.test_dataset.dictionary = self.dictionary

    def init_model(self,encoder=None):
        if self.usingw2v is False:
            self.model = Classifier(encoder,self.hidden_size,self.batch_size,self.embed_size,len(self.train_dataset.Tok2Ind),self.maxlength,False,self.usingswish)
        elif self.usingw2v is True:
            self.model = Classifier(encoder,self.hidden_size,self.batch_size,self.embed_size,len(self.dictionary.vocab),self.maxlength,True,self.usingswish)
        if self.loaddictpath is not None:
            self.model.load_state_dict(torch.load(self.loaddictpath))
        if self.cuda is True:
            self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            total, correct, avgloss = self.train_util()
            print('Epoch [%d/%d], Loss: %.4f, Acc: %.4f'%(epoch+1,self.num_epochs,avgloss/total,correct/total),file=self.Log)
            if self.valddata is not None:
                self.vald()
            self.Log.flush()
            if epoch%20==19 or epoch==self.num_epochs-1:
                self.checkpoint()

    def train_util(self):
        self.model.train()
        total = 0
        correct = 0
        avgloss = 0
        for i,(inputs,labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            for j in range(7):
                inputs[j] = Variable(inputs[j])
            if self.cuda is True:
                for j in range(7):
                    inputs[j] = inputs[j].cuda()
                labels = labels.cuda()
            outputs = self.model(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6])
            _, predicted = torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=int((predicted==labels).sum())
            loss = self.criterion(outputs,Variable(labels))
            avgloss+=float(loss.data.cpu())*labels.size(0)
            loss.backward()
            self.optimizer.step()
            if i%100==99:
                print('STEP %d, Loss: %.4f, Acc: %.4f'%(i+1,loss.data.cpu(),int((predicted==labels).sum())/labels.size(0)*100),file=self.Log)
            self.Log.flush()
        return total,correct,avgloss

    def checkpoint(self):
        if self.savedictpath is not None:
            torch.save(self.model.state_dict(),self.savedictpath)

    def vald(self):
        self.model.eval()
        total = 0
        correct = 0
        for inputs,labels in self.valdloader:
            for j in range(7):
                inputs[j] = Variable(inputs[j])
            if self.cuda is True:
                for j in range(7):
                    inputs[j] = inputs[j].cuda()
                labels = labels.cuda()
            outputs = self.model(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6])
            _, predicted = torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=int((predicted==labels).sum())
        print('-----------VALDACC : %.4f-----------'%(correct/total*100),file=self.Log)

    def test(self):
        self.model.eval()
        ans = []
        for inputs in self.testloader:
            for j in range(7):
                inputs[j] = Variable(inputs[j])
            if self.cuda is True:
                for j in range(7):
                    inputs[j] = inputs[j].cuda()
            outputs = self.model(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6])
            _, predicted = torch.max(outputs.data,1)
            ans+=predicted.cpu().numpy().tolist()
        ans = np.asarray(ans)
        print('id,ans',file=self.result)
        for i in range(len(ans)):
            print('%d,%d'%(i,ans[i]),file=self.result)


def main():
    Model = Frame(traindata=None,
                  extradata=None,
                  valddata=None,
                  testdata='./dataset/testing_data_seg.csv',
                  maxlength=30,
                  batch_size=100,
                  hidden_size=100,
                  embed_size=50,
                  num_epochs=500,
                  learning_rate=5e-4,
                  usingw2v = True,
                  trainw2v = False,
                  w2vwindow=5,
                  usingswish=True,
                  cuda=True,
                  Log='LSTMACC',
                  loaddictpath='BEST.plk',
                  savedictpath='LSTM.plk',
                  resultpath='ans.csv')
    if Model.traindata is not None:
        Model.loadtraindata()
    if Model.extradata is not None:
        Model.loadextradata(Model.train_dataset.dictionary)
    if Model.valddata is not None:
        Model.loadvalddata(Model.train_dataset.dictionary)
    if Model.testdata is not None:
        if Model.traindata is not None:
            Model.loadtestdata(Model.train_dataset.dictionary)
        else:
            Model.loadtestdata(None)

    if Model.trainw2v is not False:
        Model.trainword2vec(Model.train_dataset.df+Model.extra_dataset.df)
    if Model.usingw2v is not False:
        Model.loadword2vec()

    Model.init_model(GRUencoder)

    if Model.traindata is not None:
        Model.train()
    if Model.testdata is not None:
        Model.test()

main()
