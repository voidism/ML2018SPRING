from train import *

acc_result = [['GD method','Matrix method']]
s = 3
for i in range(1,s+1):
    print("===Validation Block(%d/%d)===" % (i,s))
    x,y,vx,vy = load_data(False,s,i)
    #print("GD method ...")
    #train(False,s,i,0)
    #B_T = valid(vx,vy)

    print('Matrix method ...')
    w = np.zeros(len(x[0]))
    w = np.matmul(np.matmul(inv(np.matmul(x.T, x)), x.T), y)
    np.save('model_w.npy', w)
    B_T = valid(x,y)
    B_W = valid(vx, vy)

    acc_result.append([B_T,B_W])

for i in acc_result:
    print(i)

