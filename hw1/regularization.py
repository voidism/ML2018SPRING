from train import *

acc_result = []
s = 5
for j in range(30):
    temp_res = np.array([[0.0,0.0,0.0]])
    for i in range(1,s+1):
        lamb = 1000+j*6#(10**(j/20+2))
        print("===Validation Block(%d/%d)===" % (i,s))
        x,y,vx,vy = load_data(False,s,i)
        print("Training Set ...")
        w = np.zeros(len(x[0]))
        w = np.matmul(np.matmul(inv(np.matmul(x.T, x) + lamb * np.identity(len(x[0]))), x.T), y)
        np.save('model_w.npy', w)
        #train(0,s,i,0,0,lamb)
        B_T = valid(x,y)

        print('Valid Set ...')
        B_W = valid(vx, vy)

        temp_res+=np.array([[lamb,B_T,B_W]])
    acc_result.append(temp_res/5)

for i in acc_result:
    print(i)

acc_result = np.array(acc_result)
np.save('reg_acc_0-9.npy',acc_result)
print(acc_result.T[0][0])
print(acc_result.T[1][0])
print(acc_result.T[2][0])
plt.semilogx(
    acc_result.T[0][0],
    acc_result.T[1][0],
    acc_result.T[0][0],
    acc_result.T[2][0]
)
# for i,j in zip(acc_result.T[0][0],acc_result.T[1][0]):
#     plt.text(i,j,str(j))
# for i,j in zip(acc_result.T[0][0],acc_result.T[2][0]):
#     plt.text(i, j, str(j))
plt.legend(['Training','Validation'])
plt.ylabel('ave cost')
plt.xlabel('lambda')
plt.title('Compare between diff regularization')
plt.savefig('reg_MT500-1500.png')

