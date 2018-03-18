import csv, sys
import numpy as np
import json
import math
import random
from math import log, floor
from numpy.linalg import inv
import time
import pandas as pd
import sys


global feature_dict
# best -->
feature_dict = [0, 0,0,0,0,2, 0,2,9,9,2,
                  0,2,0,0,0, 0,0,9]

def preprocess(filename):
    f = open(filename)
    r =csv.reader(f)
    l = list(r)
    new = []
    for i in l:
        if i[1] == 'RAINFALL':
            temprow = []
            for j in i[2:]:
                if j == "NR":
                    temprow.append(0.0)
                else:
                    temprow.append(float(j))
            new.append(np.array(temprow))
        else:
            temprow = np.array(i[2:],dtype = float)
            tf = temprow<=0
            if True in tf:
                if False not in tf:
                    pass
                    #print(l.index(i),temprow)
                else:
                    #print(l.index(i), temprow)
                    if temprow[-2]==11 and temprow[-3]==8 and temprow[-4] == 5:
                        temprow[-1] = 14
                    elif temprow[-2]==24 and temprow[-3]==31 and temprow[-4] == 40:
                        temprow[-1] = 22
                    elif temprow[-1]==0.3 and temprow[-2]==0 and temprow[-3] == 11:
                        pass
                    elif temprow[0]==45 and temprow[1]==77 and temprow[2] == 83:
                        temprow[-2]=78
                        temprow[-3]=95
                    else:
                        temprow = np.where(temprow<-1, -temprow, temprow)
                        temprow = np.where(((temprow>=-1)&(temprow<=0)),np.nan,temprow)
                        s = pd.Series(temprow)
                        s = s.interpolate()
                        temprow = np.array(s)

                        if True in np.isnan(temprow):
                            idx = np.argmax(np.isnan(temprow)==False)
                            temprow = np.where(np.isnan(temprow)==True, temprow[idx], temprow)
                        #print(l.index(i), temprow)
            new.append(temprow)

    return new



def test_to_matrix(filename):
    global feature_dict
    f = feature_dict
    traindata = preprocess(filename)
    topdata = []
    idx = 0
    for i in range(260):
        data = []
        data.append(traindata[18 * i + 0][9-f[0]:])
        data.append(traindata[18 * i + 1][9-f[1]:])
        data.append(traindata[18 * i + 2][9-f[2]:])
        data.append(traindata[18 * i + 3][9-f[3]:])
        data.append(traindata[18 * i + 4][9-f[4]:])
        data.append(traindata[18 * i + 5][9-f[5]:])
        data.append(traindata[18 * i + 6][9-f[6]:])
        data.append(traindata[18 * i + 7][9-f[7]:])
        data.append(traindata[18 * i + 8][9-f[8]:])
        data.append(traindata[18 * i + 9][9-f[9]:])

        data.append((traindata[18 * i + 9][9 - f[18]:])**2)

        data.append(traindata[18 * i + 10][9-f[10]:])
        data.append(traindata[18 * i + 11][9-f[11]:])
        data.append(traindata[18 * i + 12][9-f[12]:])
        data.append(traindata[18 * i + 13][9-f[13]:])
        data.append(traindata[18 * i + 14][9-f[14]:])
        data.append(traindata[18 * i + 15][9-f[15]:])
        data.append(traindata[18 * i + 16][9-f[16]:])
        data.append(traindata[18 * i + 17][9-f[17]:])

        data.append(np.array([1.0]))
        data = np.concatenate(data)
        topdata.append(data)
        idx+=1
    return topdata

if __name__ == '__main__':
    if sys.argv[3]=="--best":
        w = np.load('model_w_HW1BEST.npy')
    else:
        w = np.load('model_w_HW1.npy')
    tests = test_to_matrix(sys.argv[1])
    idx = 0
    ans = []
    for i in tests:
        x = np.array(i,dtype=float)
        res = w.dot(x)
        #res = np.absolute(res)
        if res < 0:
            res = 0.0
        ans.append(["id_"+str(idx),res])
        idx+=1

    filename = sys.argv[2]
    text = open(filename, "w")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    s.writerow(["id", "value"])
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()