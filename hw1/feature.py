#from train import feature_dict
#from train import *
import csv, sys
import numpy as np
import json
import math
import random
from math import log, floor
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time
import train as t

acc_result = []
s = 5
#tempx, _y, _vx, _vy = load_data()
topic = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10',
         'PM2.5','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC',
         'WIND_SPEED','WS_HR','PM2.5^2']

#global feature_dict
for f in range(len(t.feature_dict)):
    for j in range(10):
        print("Examinate feature: ",topic[f],"for",j,"HRs")
        #t.feature_dict = [9,  9, 9, 9, 9, 9,  9, 9, 9, 9, 9,
                              #9, 9, 9, 9, 9,  9, 9, 9]
        t.feature_dict[f] = j
        temp_res = np.array([[0.0,0.0,0.0]])
        for i in range(1,s+1):
            x,y,vx,vy = t.load_data(False,s,i)
            print("===Validation Block(%d/%d)===" % (i,s))
            print("Training Set ...")
            w = np.zeros(len(x[0]))
            w = np.matmul(np.matmul(inv(np.matmul(x.T, x)), x.T), y)
            np.save('model_w.npy', w)
            B_T = t.valid(x,y)

            print('Valid Set ...',"in dimension: ",len(x[0]))
            B_W = t.valid(vx, vy)

            temp_res += np.array([[j,B_T,B_W]])
        acc_result.append(temp_res/5)

for i in acc_result:
    print(i)

acc_result = np.array(acc_result)
np.save('feature_hr.npy',acc_result)
print(acc_result.T[0][0])
print(acc_result.T[1][0])
print(acc_result.T[2][0])
# plt.plot(
#     acc_result.T[0][0],
#     acc_result.T[1][0],
#     acc_result.T[0][0],
#     acc_result.T[2][0]
# )
# for i,j in zip(acc_result.T[0][0],acc_result.T[1][0]):
#     plt.text(i,j,str(j))
# for i,j in zip(acc_result.T[0][0],acc_result.T[2][0]):
#     plt.text(i, j, str(j))
# plt.legend(['Training','Validation'])
# plt.ylabel('ave cost')
# plt.xlabel('ignore term')
# plt.title('Compare between diff feature HRs selected')
# plt.savefig('feature_HR.png')



for i in range(len(topic)):
    #print(acc_result.T[1][0][0] + acc_result.T[1][0][9 * i + 1:9 * i + 11],list(range(-1,10)))
    plt.scatter(
        list(range(10)),
        acc_result.T[1][0][10*i:10*i+10])
    plt.scatter(list(range(10)),
        acc_result.T[2][0][10 * i:10 * i + 10])
    plt.legend(['Training','Validation'])
    plt.axhline(y=acc_result.T[1][0][10*i+9], color='blue', linestyle='-')
    plt.axhline(y=acc_result.T[2][0][10*i+9], color='orange', linestyle='-')
    plt.ylabel('ave cost')
    plt.xlabel('selected HRs')
    plt.title('Compare between diff '+topic[i]+' selected HRs')
    plt.savefig('feature_HR\\feature_'+topic[i]+'_selected_HRs.png')
    plt.clf()

